/* The modem owns this entire SRAM bank while adctrig runs. Heap reservation
 * alone does not prevent a future firmware's static sections growing into it. */
ASSERT(_iram_end <= 0x40820000, "C5 RF dump overlaps IRAM")
ASSERT(_data_end <= 0x40820000, "C5 RF dump overlaps data")
ASSERT(_bss_end <= 0x40820000, "C5 RF dump overlaps BSS")
/* Above the 128 KiB modem bank. chip.h reserves the same span from the heap. */
SECTIONS {
    .c5_spectrum 0x40840000 (NOLOAD) : ALIGN(16) {
        *(.c5_spectrum)
    }
}
ASSERT(SIZEOF(.c5_spectrum) <= 0x7000, "C5 spectrum workspace exceeds the reserved slot")
