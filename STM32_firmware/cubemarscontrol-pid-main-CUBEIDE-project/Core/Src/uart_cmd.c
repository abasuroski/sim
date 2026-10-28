#include "uart_cmd.h"
#include "stm32f4xx_hal.h"
#include <string.h>

#define CMD_BUF_SIZE UART_CMD_LINE_MAX

static MotorState motors[NUM_MOTORS];
static float servo_pos[NUM_SERVOS] = {0.0f, 0.0f, 0.0f};
static char buf[CMD_BUF_SIZE];
static uint8_t idx = 0;
static volatile uint8_t e_stop_pending = 0;

// atof() used to turn any damaged value into 0.0; on 2026-10-22 M1 latched
// P = 0 and drove toward it at full gain. Damaged lines are now dropped and
// the motor keeps its previous command: value commands must carry a "*HH"
// XOR checksum, and lines hit by a UART error are rejected.
static uint8_t line_corrupt  = 0;   // a UART error hit a byte of this line
static uint8_t line_overflow = 0;   // line longer than CMD_BUF_SIZE - 1
static volatile uint32_t rejected_count = 0;
// Latest rejected line, for the main loop to report ("[UART] rejected ...").
static char rejected_buf[CMD_BUF_SIZE];
static volatile uint8_t rejected_pending = 0;

uint8_t uart_cmd_e_stop_requested(void) { return e_stop_pending; }
void    uart_cmd_clear_e_stop(void)     { e_stop_pending = 0; }

void     uart_cmd_mark_line_corrupt(void) { line_corrupt = 1; }
uint32_t uart_cmd_rejected_count(void)    { return rejected_count; }

uint8_t uart_cmd_take_rejected(char *out, uint8_t out_size)
{
    // The UART ISR writes rejected_buf; block it for the short copy.
    uint8_t taken = 0;
    __disable_irq();
    if (rejected_pending && out_size > 0) {
        strncpy(out, rejected_buf, out_size - 1);
        out[out_size - 1] = '\0';
        rejected_pending = 0;
        taken = 1;
    }
    __enable_irq();
    return taken;
}

static void reject_line(void)
{
    rejected_count++;
    memcpy(rejected_buf, buf, idx);
    rejected_buf[idx] = '\0';
    rejected_pending = 1;
}

// Strict, fast number parse for the PC's "%.5f" values: [-]digits[.digits],
// the whole string, at least one digit. Not strtod/atof: newlib's version
// is slow enough that, run here in the receive ISR, the next command's bytes
// overran the UART (2026-10-22 log: none of 100 V lines sent right after a
// P line ever reached the motor).
#define PARSE_MAX_INT_DIGITS  6
#define PARSE_MAX_FRAC_DIGITS 7
static uint8_t parse_value(const char *s, float *out)
{
    uint8_t neg = 0;
    if (*s == '-') { neg = 1; s++; }
    // Integer and fractional parts are kept apart so each fits in uint32.
    uint32_t int_part = 0, frac_part = 0, scale = 1;
    uint8_t int_digits = 0, frac_digits = 0;
    while (*s >= '0' && *s <= '9') {
        if (++int_digits > PARSE_MAX_INT_DIGITS) return 0;
        int_part = int_part * 10u + (uint32_t)(*s++ - '0');
    }
    if (*s == '.') {
        s++;
        while (*s >= '0' && *s <= '9') {
            if (++frac_digits > PARSE_MAX_FRAC_DIGITS) return 0;
            frac_part = frac_part * 10u + (uint32_t)(*s++ - '0');
            scale *= 10u;
        }
    }
    if (*s != '\0' || int_digits + frac_digits == 0) return 0;
    float v = (float)int_part + (float)frac_part / (float)scale;
    *out = neg ? -v : v;
    return 1;
}

void uart_cmd_init(void)
{
    // Motor 1: AK60, CAN ID 104
    motors[0].pos = 0.0f;
    motors[0].vel = 0.0f;
    motors[0].kp  = 3.0f;    // inner MIT spring
    motors[0].kd  = 0.2f;    // inner MIT damper
    motors[0].tff = 0.0f;
    motors[0].can_id = 104;
    motors[0].type = MOTOR_TYPE_AK60;
    motors[0].enable_pending = 0;
    motors[0].enabled = 0;
    motors[0].set_origin_pending = 0;
    motors[0].pos_offset = 0.0f;
    motors[0].pos_offset_counts = 0;
    motors[0].fb_only = 0;

    // Motor 2: AK70, CAN ID 1 (needs enable)
    motors[1].pos = 0.0f;
    motors[1].vel = 0.0f;
    motors[1].kp  = 5.0f;    // inner MIT spring
    motors[1].kd  = 0.3f;    // inner MIT damper
    motors[1].tff = 0.0f;
    motors[1].can_id = 2;
    motors[1].type = MOTOR_TYPE_AK70;
    motors[1].enable_pending = 0;
    motors[1].enabled = 0;
    motors[1].set_origin_pending = 0;
    motors[1].fb_only = 0;

    // Motor 3: AK70, CAN ID 2 (needs enable)
    motors[2].pos = 0.0f;
    motors[2].vel = 0.0f;
    motors[2].kp  = 5.0f;    // inner MIT spring
    motors[2].kd  = 0.3f;    // inner MIT damper
    motors[2].tff = 0.0f;
    motors[2].can_id = 1;
    motors[2].type = MOTOR_TYPE_AK70;
    motors[2].enable_pending = 0;
    motors[2].enabled = 0;
    motors[2].set_origin_pending = 0;
    motors[2].fb_only = 0;

    // Motor 4: AK40, CAN ID 3 (needs enable)
    motors[3].pos = 0.0f;
    motors[3].vel = 0.0f;
    motors[3].kp  = 5.0f;    // inner MIT spring
    motors[3].kd  = 0.3f;    // inner MIT damper
    motors[3].tff = 0.0f;
    motors[3].can_id = 3;
    motors[3].type = MOTOR_TYPE_AK40;
    motors[3].enable_pending = 0;
    motors[3].enabled = 0;
    motors[3].set_origin_pending = 0;
    motors[3].fb_only = 0;


    idx = 0;
}

MotorState *uart_cmd_motor(uint8_t motor_idx)
{
    if (motor_idx >= NUM_MOTORS) return &motors[0];
    return &motors[motor_idx];
}

// Copy one motor's command with the UART ISR held off, so a command line
// that lands mid-copy cannot mix old and new fields (the M command sets all
// five together).
void uart_cmd_read(uint8_t motor_idx, MotorCmd *out)
{
    const MotorState *m = uart_cmd_motor(motor_idx);
    uint32_t primask = __get_PRIMASK();
    __disable_irq();
    out->pos = m->pos;
    out->vel = m->vel;
    out->kp  = m->kp;
    out->kd  = m->kd;
    out->tff = m->tff;
    __set_PRIMASK(primask);
}

float *uart_cmd_servo_pos(void)
{
    return servo_pos;
}

static uint8_t hex_digit(char c)
{
    if (c >= '0' && c <= '9') return (uint8_t)(c - '0');
    if (c >= 'A' && c <= 'F') return (uint8_t)(c - 'A' + 10);
    if (c >= 'a' && c <= 'f') return (uint8_t)(c - 'a' + 10);
    return 0xFF;
}

// Optional "*HH" suffix: HH = XOR of every byte before the '*', in hex.
// Returns 1 if present and correct (the suffix is cut off buf), 0 if absent,
// -1 if malformed or wrong. Since UART receive moved to DMA a damaged byte
// can no longer be pinned to its line, so value commands require it.
static int8_t strip_checksum(void)
{
    char *star = memchr(buf, '*', idx);
    if (star == NULL) return 0;
    uint8_t len = (uint8_t)(star - buf);
    if (len + 3 != idx) return -1;
    uint8_t hi = hex_digit(buf[len + 1]), lo = hex_digit(buf[len + 2]);
    if (hi > 0xF || lo > 0xF) return -1;
    uint8_t sum = 0;
    for (uint8_t i = 0; i < len; i++) sum ^= (uint8_t)buf[i];
    if (sum != (uint8_t)((hi << 4) | lo)) return -1;
    buf[len] = '\0';
    idx = len;
    return 1;
}

// "<pos>,<vel>,<kp>,<kd>,<tff>": all five numbers or nothing.
static uint8_t parse_mit_frame(const char *s, float out[5])
{
    char field[CMD_BUF_SIZE];
    for (uint8_t k = 0; k < 5; k++) {
        uint8_t n = 0;
        while (s[n] != ',' && s[n] != '\0') n++;
        if ((k < 4) != (s[n] == ',')) return 0;   // comma after the first four only
        memcpy(field, s, n);
        field[n] = '\0';
        if (!parse_value(field, &out[k])) return 0;
        s += n + 1;
    }
    return 1;
}

static void parse_cmd(void)
{
    buf[idx] = '\0';

    // Global emergency stop: "0X", checksum optional. Honoured even on a
    // damaged line or with a wrong checksum, because stopping is always the
    // safe direction.
    if ((idx == 2 || (idx == 5 && buf[2] == '*')) &&
        buf[0] == '0' && (buf[1] == 'X' || buf[1] == 'x')) {
        e_stop_pending = 1;
        return;
    }

    if (line_corrupt || line_overflow) {
        if (idx > 0 || line_overflow) reject_line();
        return;
    }
    if (idx < 2) {
        // A lone fragment such as "1" is what remains of a line whose
        // command letter was lost; a bare terminator ("\r\n") is normal.
        if (idx > 0) reject_line();
        return;
    }

    uint8_t motor_num = (uint8_t)(buf[0] - '1');
    // Not addressed to a motor or servo (e.g. "PING"): ignored.
    if (motor_num >= NUM_MOTORS + NUM_SERVOS) return;

    int8_t checksum = strip_checksum();
    if (checksum < 0 || idx < 2) { reject_line(); return; }

    char cmd = buf[1];
    float val = 0.0f;

    // Servo indices: '5','6','7' → motor_num 4,5,6
    if (motor_num >= NUM_MOTORS) {
        if (cmd == 'P' || cmd == 'p') {
            if (checksum != 1 || !parse_value(&buf[2], &val)) { reject_line(); return; }
            if (val < -120.0f) val = -120.0f;
            if (val > 120.0f) val = 120.0f;
            servo_pos[motor_num - NUM_MOTORS] = val;
        }
        return;
    }

    MotorState *m = &motors[motor_num];
    float frame[5];

    switch (cmd) {
        case 'P': case 'p':
        case 'V': case 'v':
        case 'K': case 'k':
        case 'D': case 'd':
        case 'T': case 't':
            if (checksum != 1 || !parse_value(&buf[2], &val)) { reject_line(); return; }
            break;
        case 'M': case 'm':
            if (checksum != 1 || !parse_mit_frame(&buf[2], frame)) { reject_line(); return; }
            break;
        case 'E': case 'e':
        case 'R': case 'r':
        case 'O': case 'o':
            // Argument-free: trailing characters mean the line is damaged
            // (e.g. a flipped bit turned 'P' into 'E').
            if (idx != 2) { reject_line(); return; }
            break;
        default: return;   // unknown or retired (G/H/J outer PID): ignored
    }

    switch (cmd) {
        case 'P': case 'p': m->pos = val; break;
        case 'V': case 'v': m->vel = val; break;
        case 'K': case 'k': m->kp  = val; break;
        case 'D': case 'd': m->kd  = val; break;
        case 'T': case 't': m->tff = val; break;
        case 'M': case 'm':
            // Runs in the UART ISR, and the main loop copies commands with
            // uart_cmd_read(), so the five fields always change together.
            m->pos = frame[0];
            m->vel = frame[1];
            m->kp  = frame[2];
            m->kd  = frame[3];
            m->tff = frame[4];
            break;
        case 'E': case 'e':
            m->enable_pending = 1;
            m->fb_only = 0;
            break;
        case 'R': case 'r':
            m->fb_only = 1;
            m->enabled = 0;
            m->enable_pending = 0;
            break;
        case 'O': case 'o': m->set_origin_pending = 1; break;
        default: break;
    }
}

void uart_cmd_feed(uint8_t byte)
{
    if (byte == '\n' || byte == '\r') {
        parse_cmd();
        idx = 0;
        line_corrupt  = 0;
        line_overflow = 0;
    } else {
        if (idx < CMD_BUF_SIZE - 1)
            buf[idx++] = (char)byte;
        else
            line_overflow = 1;
    }
}
