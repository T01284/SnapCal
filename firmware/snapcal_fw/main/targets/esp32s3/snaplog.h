/* SNAPLOG: host-free snapshot logging to the "snaplog" flash partition, so a board can
 * capture inside a closed enclosure with no cable attached (battery power).
 *
 *   LOG ARM count n rate fmt gain delay_s gap_ms   erase, store config (current FREQ/LPF)
 *   LOG RUN                                       run now, USB attached (tethered reference)
 *   LOG?                                          state and config
 *   LOG READ i                                    "LOGDATA i n fmt start_us elapsed crc gap_us\n"+payload
 *   LOG CLEAR                                     forget config and data
 *
 * An armed, not yet started log also starts by itself once the USB SOF counter has not
 * moved for 2 s (cable pulled with a battery attached, or boot on battery; 3 s grace after boot).
 * Runs never reply: there may be no host. Flash writes happen between captures only.
 *
 * Partition layout: sector 0 = config at 0, run marker at 256, done marker at 512;
 * slot i at 4096 + i * slot_bytes = 32-byte header + packed payload.
 * Included by receiver.c after capture_raw(). */
#include "esp_partition.h"
#include "soc/usb_serial_jtag_struct.h"

#define SNAPLOG_MAGIC 0x474c4e53u /* "SNLG" */
#define SNAPLOG_RUN 0x4e555253u   /* "SRUN" */
#define SNAPLOG_DONE 0x454e4f44u  /* "DONE" */
#define SNAPLOG_SLOT 0x50414e53u  /* "SNAP" */
#define SNAPLOG_SUBTYPE 0x40

typedef struct {
    uint32_t magic, version, count, n, rate, fmt, freq, gain, delay_s, gap_ms, slot_bytes;
    int32_t filter;
} snaplog_cfg_t;
typedef struct { uint32_t magic, tethered; int64_t t_us; } snaplog_run_t;
typedef struct { uint32_t magic, done; } snaplog_done_t;
typedef struct {
    uint32_t magic, index, n, fmt;
    int64_t start_us;
    uint32_t elapsed_us, crc, gap_us, bytes;
} snaplog_slot_t;
_Static_assert(sizeof(snaplog_slot_t) == 40, "slot header layout");
#define SNAPLOG_HDR 64u

static const esp_partition_t *snaplog_part(void) {
    return esp_partition_find_first(ESP_PARTITION_TYPE_DATA, (esp_partition_subtype_t)SNAPLOG_SUBTYPE, "snaplog");
}
static bool snaplog_cfg(const esp_partition_t *p, snaplog_cfg_t *c) {
    return p && esp_partition_read(p, 0, c, sizeof(*c)) == ESP_OK && c->magic == SNAPLOG_MAGIC && c->version == 1;
}
static uint32_t snaplog_capacity(const esp_partition_t *p, uint32_t slot) {
    return p && slot ? (uint32_t)((p->size - 4096u) / slot) : 0u;
}
static bool snaplog_pending;
static int64_t snaplog_next_check;
static uint8_t snaplog_no_sof;
static uint32_t snaplog_last_sof = 0xffffffffu;

static bool snaplog_started(const esp_partition_t *p) {
    uint32_t m;
    return esp_partition_read(p, 256, &m, 4) == ESP_OK && m == SNAPLOG_RUN;
}
static void snaplog_refresh(void) {
    const esp_partition_t *p = snaplog_part();
    snaplog_cfg_t c;
    snaplog_pending = snaplog_cfg(p, &c) && !snaplog_started(p);
    /* USB enumeration takes about a second after boot; do not mistake it for no host. */
    snaplog_next_check = esp_timer_get_time() + 3000000;
}

static void snaplog_run(bool tethered) {
    const esp_partition_t *p = snaplog_part();
    snaplog_cfg_t c;
    snaplog_pending = false;
    if (!snaplog_cfg(p, &c) || snaplog_started(p)) return;
    snaplog_run_t r = {SNAPLOG_RUN, tethered, esp_timer_get_time()};
    if (esp_partition_write(p, 256, &r, sizeof(r)) != ESP_OK) return;
    frequency_mhz = c.freq;
    rx_filter = c.filter;
    gain_mode = GAIN_MANUAL;
    gain_code = c.gain;
    snap_jitter_us = 0;
    gain_reconfigure();
    for (uint32_t s = 0; s < c.delay_s; s++) vTaskDelay(pdMS_TO_TICKS(1000));
    uint32_t done = 0;
    for (uint32_t i = 0; i < c.count; i++) {
        /* At least one tick per snapshot keeps the idle task (watchdog) running; the
         * sub-millisecond remainder decouples capture starts from the tick grid. */
        uint32_t gap = c.gap_ms ? esp_random() % (c.gap_ms * 1000u + 1u) : 0u;
        vTaskDelay(pdMS_TO_TICKS(gap / 1000u) + 1);
        esp_rom_delay_us(gap % 1000u);
        size_t bytes = 0;
        uint32_t elapsed = 0, crc = 0;
        bool ok = false;
        for (int k = 0; k < 3 && !ok; k++) ok = capture_raw(c.n, c.rate, c.fmt, true, &bytes, &elapsed, &crc);
        if (!ok) break;
        snaplog_slot_t h = {SNAPLOG_SLOT, i, c.n, c.fmt, snap_start_us, elapsed, crc, gap, (uint32_t)bytes};
        size_t off = 4096u + (size_t)i * c.slot_bytes;
        if (esp_partition_write(p, off, &h, sizeof(h)) != ESP_OK ||
            esp_partition_write(p, off + SNAPLOG_HDR, IQ_BUFFER, bytes) != ESP_OK) break;
        done++;
    }
    snaplog_done_t d = {SNAPLOG_DONE, done};
    (void)esp_partition_write(p, 512, &d, sizeof(d));
}

/* Called from the idle command loop: start an armed log once USB SOFs stop. */
static void snaplog_poll(void) {
    if (!snaplog_pending) return;
    int64_t now = esp_timer_get_time();
    if (now < snaplog_next_check) return;
    snaplog_next_check = now + 500000;
    uint32_t sof = USB_SERIAL_JTAG.fram_num.sof_frame_index;
    snaplog_no_sof = sof == snaplog_last_sof ? snaplog_no_sof + 1 : 0;
    snaplog_last_sof = sof;
    if (snaplog_no_sof >= 4) snaplog_run(false);
}

static bool snaplog_command(const char *line) {
    if (strncmp(line, "LOG", 3)) return false;
    const esp_partition_t *p = snaplog_part();
    unsigned count, n, rate, fmt, gain, delay_s, gap_ms, i;
    char extra, h[200];
    snaplog_cfg_t c;
    if (!strcmp(line, "LOG?")) {
        if (!p) { reply("LOG NONE\n"); return true; }
        if (!snaplog_cfg(p, &c)) {
            snprintf(h, sizeof(h), "LOG EMPTY %" PRIu32 "\n", (uint32_t)p->size);
            reply(h); return true;
        }
        snaplog_run_t r = {0}; snaplog_done_t d = {0};
        (void)esp_partition_read(p, 256, &r, sizeof(r));
        (void)esp_partition_read(p, 512, &d, sizeof(d));
        const char *st = d.magic == SNAPLOG_DONE ? "DONE" : r.magic == SNAPLOG_RUN ? "RUNNING" : "ARMED";
        snprintf(h, sizeof(h), "LOG %s %" PRIu32 " %" PRIu32 " %" PRIu32 " %" PRIu32 " %" PRIu32 " %" PRIu32
                 " %" PRIu32 " %" PRIu32 " %" PRId32 " %" PRIu32 " %" PRIu32 " %" PRIu32 " %" PRIu32 "\n",
                 st, c.count, c.n, c.rate, c.fmt, c.freq, c.gain, c.delay_s, c.gap_ms, c.filter, c.slot_bytes,
                 snaplog_capacity(p, c.slot_bytes), d.magic == SNAPLOG_DONE ? d.done : 0u,
                 r.magic == SNAPLOG_RUN ? r.tethered : 0u);
        reply(h); return true;
    }
    if (!p) { reply("ERR nolog\n"); return true; }
    if (!strcmp(line, "LOG CLEAR")) {
        reply(esp_partition_erase_range(p, 0, 4096) == ESP_OK ? "OK\n" : "ERR erase\n");
        snaplog_pending = false; return true;
    }
    if (sscanf(line, "LOG ARM %u %u %u %u %u %u %u %c", &count, &n, &rate, &fmt, &gain, &delay_s, &gap_ms, &extra) == 7) {
        if (!count || n < 256 || n > IQ_WORDS || (rate != 0 && rate != 1 && rate != 6) ||
            (fmt != 0 && fmt != 16 && fmt != 20) || gain > gain_max() || delay_s > 3600 || gap_ms > 60000) {
            reply("ERR args\n"); return true;
        }
        uint32_t slot = (uint32_t)((SNAPLOG_HDR + wire_size(n, fmt) + 4095u) & ~4095u);
        if (count > snaplog_capacity(p, slot)) { reply("ERR capacity\n"); return true; }
        if (esp_partition_erase_range(p, 0, 4096u + (size_t)count * slot) != ESP_OK) { reply("ERR erase\n"); return true; }
        c = (snaplog_cfg_t){SNAPLOG_MAGIC, 1, count, n, rate, fmt, frequency_mhz, gain, delay_s, gap_ms, slot, rx_filter};
        if (esp_partition_write(p, 0, &c, sizeof(c)) != ESP_OK) { reply("ERR write\n"); return true; }
        snaplog_pending = true;
        snaplog_no_sof = 0;
        snaplog_last_sof = 0xffffffffu;
        snprintf(h, sizeof(h), "OK %" PRIu32 " %" PRIu32 "\n", slot, snaplog_capacity(p, slot));
        reply(h); return true;
    }
    if (!strcmp(line, "LOG RUN")) {
        if (!snaplog_cfg(p, &c) || snaplog_started(p)) { reply("ERR state\n"); return true; }
        reply("OK\n");
        vTaskDelay(pdMS_TO_TICKS(20));
        snaplog_run(true);
        return true;
    }
    if (sscanf(line, "LOG READ %u %c", &i, &extra) == 1) {
        snaplog_slot_t s;
        if (!snaplog_cfg(p, &c) || i >= c.count ||
            esp_partition_read(p, 4096u + (size_t)i * c.slot_bytes, &s, sizeof(s)) != ESP_OK ||
            s.magic != SNAPLOG_SLOT || s.bytes > IQ_WORDS * 4u) {
            reply("ERR slot\n"); return true;
        }
        if (esp_partition_read(p, 4096u + (size_t)i * c.slot_bytes + SNAPLOG_HDR, IQ_BUFFER, s.bytes) != ESP_OK) {
            reply("ERR read\n"); return true;
        }
        snprintf(h, sizeof(h), "LOGDATA %" PRIu32 " %" PRIu32 " %" PRIu32 " %" PRId64 " %" PRIu32 " %08" PRIx32 " %" PRIu32 "\n",
                 s.index, s.n, s.fmt, s.start_us, s.elapsed_us, s.crc, s.gap_us);
        (void)(send_bytes(h, strlen(h)) && send_bytes(IQ_BUFFER, s.bytes));
        return true;
    }
    reply("ERR command\n");
    return true;
}
