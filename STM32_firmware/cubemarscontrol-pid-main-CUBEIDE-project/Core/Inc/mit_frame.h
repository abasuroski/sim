#ifndef MIT_FRAME_H
#define MIT_FRAME_H

#include <stdint.h>
#include "uart_cmd.h"   // MotorType, MotorCmd

// CubeMars MIT command ranges. Shared by the CAN packing in main.c and the
// "A" command decoder in uart_cmd.c, which receives frames the PC packed.

// AK60 parameter limits
#define AK60_P_MIN    -12.56f
#define AK60_P_MAX     12.56f
#define AK60_V_MIN    -60.0f
#define AK60_V_MAX     60.0f
#define AK60_T_MIN    -12.0f
#define AK60_T_MAX     12.0f

// AK70/80 parameter limits
#define AK70_P_MIN    -12.5f
#define AK70_P_MAX     12.5f
#define AK70_V_MIN    -30.0f
#define AK70_V_MAX     30.0f
#define AK70_T_MIN    -18.0f
#define AK70_T_MAX     18.0f

// AK40 parameter limits
#define AK40_P_MIN    -12.5f
#define AK40_P_MAX     12.5f
#define AK40_V_MIN    -45.5f
#define AK40_V_MAX     45.5f
#define AK40_T_MIN    -5.0f
#define AK40_T_MAX     5.0f

// Shared limits
#define KP_MIN    0.0f
#define KP_MAX    500.0f
#define KD_MIN    0.0f
#define KD_MAX    5.0f

float mit_uint_to_float(int x_int, float x_min, float x_max, int bits);

// Pack one MIT command into the 8 CAN data bytes for this motor type.
// AK60: KP/KD/Pos/Vel/Torque byte order. AK70/80 and AK40: Pos/Vel/KP/KD/Torque.
void mit_pack_cmd(MotorType type, const MotorCmd *c, float pos_offset, uint8_t out[8]);

// The inverse of mit_pack_cmd (pos_offset 0). Packing the result again gives
// the same 8 bytes, because mit_pack_cmd rounds to the nearest count.
void mit_unpack_cmd(MotorType type, const uint8_t in[8], MotorCmd *out);

#endif
