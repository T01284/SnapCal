/* On-chip mean-power energy detect for CAPE* (paper mode B: upload hits only).
 * Wire codes match spectrum_power_code (~0.5 dB/step). Hit when code >= thresh. */
#pragma once
#include <stdint.h>
#include "spectrum_math.h"

static inline int32_t energy_sext10(uint32_t field) {
    return (int32_t)((field & 0x3ffu) ^ 0x200u) - 0x200;
}

/* Mean |z|^2 over n raw NBC IQ words (I bits 0..9, Q bits 10..19). */
static inline uint32_t energy_mean_power(const uint32_t *iq, unsigned n) {
    uint64_t sum = 0;
    for(unsigned j = 0; j < n; j++) {
        uint32_t w = iq[j];
        int32_t i = energy_sext10(w), q = energy_sext10(w >> 10);
        sum += (uint32_t)(i * i) + (uint32_t)(q * q);
    }
    return n ? (uint32_t)(sum / n) : 0;
}

static inline uint8_t energy_power_code(uint32_t mean_power) {
    return spectrum_power_code(mean_power);
}
