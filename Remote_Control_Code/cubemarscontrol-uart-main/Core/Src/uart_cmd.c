#include "uart_cmd.h"
#include <stdlib.h>
#include <string.h>

#define CMD_BUF_SIZE 32

static MotorState motors[NUM_MOTORS];
static float servo_pos[NUM_SERVOS] = {0.0f, 0.0f, 0.0f};
static char buf[CMD_BUF_SIZE];
static uint8_t idx = 0;

void uart_cmd_init(void)
{
    // Motor 1: AK60, CAN ID 104
    motors[0].pos = 0.0f;
    motors[0].vel = 0.0f;
    motors[0].kp  = 2.0f;
    motors[0].kd  = 1.0f;
    motors[0].tff = 0.0f;
    motors[0].can_id = 104;
    motors[0].type = MOTOR_TYPE_AK60;
    motors[0].enable_pending = 0;
    motors[0].enabled = 0;
    motors[0].set_origin_pending = 0;
    motors[0].pos_offset = 0.0f;

    // Motor 2: AK70, CAN ID 1 (needs enable)
    motors[1].pos = 0.0f;
    motors[1].vel = 0.0f;
    motors[1].kp  = 6.0f;
    motors[1].kd  = 0.2f;
    motors[1].tff = 0.0f;
    motors[1].can_id = 1;
    motors[1].type = MOTOR_TYPE_AK70;
    motors[1].enable_pending = 0;
    motors[1].enabled = 0;
    motors[1].set_origin_pending = 0;

    // Motor 3: AK70, CAN ID 2 (needs enable)
    motors[2].pos = 0.0f;
    motors[2].vel = 0.0f;
    motors[2].kp  = 6.0f;
    motors[2].kd  = 0.2f;
    motors[2].tff = 0.0f;
    motors[2].can_id = 2;
    motors[2].type = MOTOR_TYPE_AK70;
    motors[2].enable_pending = 0;
    motors[2].enabled = 0;
    motors[2].set_origin_pending = 0;

    // Motor 4: AK40, CAN ID 3 (needs enable)
    motors[3].pos = 0.0f;
    motors[3].vel = 0.0f;
    motors[3].kp  = 6.0f;
    motors[3].kd  = 0.2f;
    motors[3].tff = 0.0f;
    motors[3].can_id = 3;
    motors[3].type = MOTOR_TYPE_AK40;
    motors[3].enable_pending = 0;
    motors[3].enabled = 0;
    motors[3].set_origin_pending = 0;

    idx = 0;
}

MotorState *uart_cmd_motor(uint8_t motor_idx)
{
    if (motor_idx >= NUM_MOTORS) return &motors[0];
    return &motors[motor_idx];
}

float *uart_cmd_servo_pos(void)
{
    return servo_pos;
}

static void parse_cmd(void)
{
    if (idx < 2) return;
    buf[idx] = '\0';

    uint8_t motor_num = (uint8_t)(buf[0] - '1');
    char cmd = buf[1];
    float val = (idx > 2) ? (float)atof(&buf[2]) : 0.0f;

    // Servo indices: '5','6','7' → motor_num 4,5,6
    if (motor_num >= NUM_MOTORS && motor_num < NUM_MOTORS + NUM_SERVOS) {
        if (cmd == 'P' || cmd == 'p') {
            if (val < -120.0f) val = -120.0f;
            if (val > 120.0f) val = 120.0f;
            servo_pos[motor_num - NUM_MOTORS] = val;
        }
        return;
    }

    if (motor_num >= NUM_MOTORS) return;

    MotorState *m = &motors[motor_num];

    switch (cmd) {
        case 'P': case 'p': m->pos = val; break;
        case 'V': case 'v': m->vel = val; break;
        case 'K': case 'k': m->kp  = val; break;
        case 'D': case 'd': m->kd  = val; break;
        case 'T': case 't': m->tff = val; break;
        case 'E': case 'e': m->enable_pending = 1; break;
        case 'O': case 'o': m->set_origin_pending = 1; break;
        case 'I': case 'i': m->can_id = (uint8_t)val; break;
        case 'A': case 'a':
            if ((int)val == 70)
                m->type = MOTOR_TYPE_AK70;
            else if ((int)val == 40)
                m->type = MOTOR_TYPE_AK40;
            else
                m->type = MOTOR_TYPE_AK60;
            m->enabled = 0;
            break;
        default: break;
    }
}

void uart_cmd_feed(uint8_t byte)
{
    if (byte == '\n' || byte == '\r') {
        parse_cmd();
        idx = 0;
    } else {
        if (idx < CMD_BUF_SIZE - 1)
            buf[idx++] = (char)byte;
    }
}
