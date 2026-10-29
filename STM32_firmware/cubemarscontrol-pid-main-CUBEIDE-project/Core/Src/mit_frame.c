#include "mit_frame.h"

static unsigned int float_to_uint(float x, float x_min, float x_max, unsigned int bits)
{
    float span = x_max - x_min;
    if (x < x_min) x = x_min;
    else if (x > x_max) x = x_max;
    // Round to the nearest count (not truncate), so a value decoded from a
    // packed frame packs back to exactly the same count.
    return (unsigned int)((x - x_min) / span * (float)((1u << bits) - 1) + 0.5f);
}

float mit_uint_to_float(int x_int, float x_min, float x_max, int bits)
{
    float span = x_max - x_min;
    return ((float)x_int) * span / ((float)((1 << bits) - 1)) + x_min;
}

typedef struct {
    float p_min, p_max, v_min, v_max, t_min, t_max;
} MitRanges;

static MitRanges ranges_for(MotorType type)
{
    MitRanges r;
    if (type == MOTOR_TYPE_AK60) {
        r.p_min = AK60_P_MIN; r.p_max = AK60_P_MAX;
        r.v_min = AK60_V_MIN; r.v_max = AK60_V_MAX;
        r.t_min = AK60_T_MIN; r.t_max = AK60_T_MAX;
    } else if (type == MOTOR_TYPE_AK40) {
        r.p_min = AK40_P_MIN; r.p_max = AK40_P_MAX;
        r.v_min = AK40_V_MIN; r.v_max = AK40_V_MAX;
        r.t_min = AK40_T_MIN; r.t_max = AK40_T_MAX;
    } else {
        r.p_min = AK70_P_MIN; r.p_max = AK70_P_MAX;
        r.v_min = AK70_V_MIN; r.v_max = AK70_V_MAX;
        r.t_min = AK70_T_MIN; r.t_max = AK70_T_MAX;
    }
    return r;
}

void mit_pack_cmd(MotorType type, const MotorCmd *c, float pos_offset, uint8_t out[8])
{
    MitRanges r = ranges_for(type);
    unsigned int p_int  = float_to_uint(c->pos + pos_offset, r.p_min, r.p_max, 16);
    unsigned int v_int  = float_to_uint(c->vel, r.v_min, r.v_max, 12);
    unsigned int kp_int = float_to_uint(c->kp,  KP_MIN, KP_MAX, 12);
    unsigned int kd_int = float_to_uint(c->kd,  KD_MIN, KD_MAX, 12);
    unsigned int t_int  = float_to_uint(c->tff, r.t_min, r.t_max, 12);

    if (type == MOTOR_TYPE_AK60) {
        out[0] = (kp_int >> 4) & 0xFF;
        out[1] = ((kp_int & 0xF) << 4) | ((kd_int >> 8) & 0xF);
        out[2] = kd_int & 0xFF;
        out[3] = (p_int >> 8) & 0xFF;
        out[4] = p_int & 0xFF;
        out[5] = (v_int >> 4) & 0xFF;
        out[6] = ((v_int & 0xF) << 4) | ((t_int >> 8) & 0xF);
        out[7] = t_int & 0xFF;
        return;
    }
    out[0] = p_int >> 8;
    out[1] = p_int & 0xFF;
    out[2] = v_int >> 4;
    out[3] = ((v_int & 0xF) << 4) | (kp_int >> 8);
    out[4] = kp_int & 0xFF;
    out[5] = kd_int >> 4;
    out[6] = ((kd_int & 0xF) << 4) | (t_int >> 8);
    out[7] = t_int & 0xFF;
}

void mit_unpack_cmd(MotorType type, const uint8_t in[8], MotorCmd *out)
{
    MitRanges r = ranges_for(type);
    int p_int, v_int, kp_int, kd_int, t_int;
    if (type == MOTOR_TYPE_AK60) {
        kp_int = (in[0] << 4) | (in[1] >> 4);
        kd_int = ((in[1] & 0xF) << 8) | in[2];
        p_int  = (in[3] << 8) | in[4];
        v_int  = (in[5] << 4) | (in[6] >> 4);
        t_int  = ((in[6] & 0xF) << 8) | in[7];
    } else {
        p_int  = (in[0] << 8) | in[1];
        v_int  = (in[2] << 4) | (in[3] >> 4);
        kp_int = ((in[3] & 0xF) << 8) | in[4];
        kd_int = (in[5] << 4) | (in[6] >> 4);
        t_int  = ((in[6] & 0xF) << 8) | in[7];
    }
    out->pos = mit_uint_to_float(p_int,  r.p_min, r.p_max, 16);
    out->vel = mit_uint_to_float(v_int,  r.v_min, r.v_max, 12);
    out->kp  = mit_uint_to_float(kp_int, KP_MIN, KP_MAX, 12);
    out->kd  = mit_uint_to_float(kd_int, KD_MIN, KD_MAX, 12);
    out->tff = mit_uint_to_float(t_int,  r.t_min, r.t_max, 12);
}
