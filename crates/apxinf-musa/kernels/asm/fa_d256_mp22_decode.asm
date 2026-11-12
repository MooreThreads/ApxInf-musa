
    .text
    .protected  mate_fa_d256_mp22
    .globl  mate_fa_d256_mp22
    .p2align    8
    .type   mate_fa_d256_mp22,@function
mate_fa_d256_mp22:
[NswD---] BIT.LUT.AND R0, AR0, SH43
[NwsD---] BIT.LUT.XOR R19, R0, C33
[NswD---] BIT.LUT.XOR R20, R0, C45
[NwsD---] BIT.LUT.XOR R21, R0, C61
[NswD---] BIT.LUT.XOR R22, R0, SH39
[N--D--s] MOV.LD.LM.B32 R5, LM4
[N--D--s] MOV.LD.LM.B32 R6, LM5
[NwwDwww] CTRL.NOP
[N--D--s] MOV.LD.LM.B32 R23, LM6
[NwwDwww] CTRL.NOP
[NwsD---] BIT.ROT.LSL R8, C45, R23
[NswD---] INT.ADD R16.s32, R8.s32, C61.s32
[NwwDs--] DMA.LD.B32 R24, SH[16:17], R8, _, _, _, _, SH[16:18]
[N--Ds--] DMA.LD.B32 R25, SH[16:17], R16, _, _, _, _, SH[16:18]
[N--Ds--] DMA.LD.B32 R26, SH[20:21], R8, _, _, _, _, SH[20:22]
[NwwDwww] CTRL.NOP
[NwsD---] BIT.ROT.LSL R24, C50, R24
[NswD---] BIT.ROT.LSL R25, C50, R25
[NwsD---] BIT.LUT.AND R12, R6, C50
[NswD---] BIT.ROT.LSL R12, C79, R12
[NwsD---] BIT.ROT.LSR R6, C45, R6
[NswD---] FOP.MOV R27, C0
[NwsD---] INT.MADD R27.s32, SH25.s32, R27.s32, R23.s32
[NswD---] BIT.ROT.LSL R5, C80, R5
[NwsD---] INT.ADD R5.s32, R5.s32, R24.s32
[NswD---] INT.BYP.TGE p0, R5.s32, R25.s32
[NwwD---] CTRL.BR.ALLP0T FINISH
[NwsD---] BIT.AND.LSR R1, C45, SH40, R0
[NswD---] BIT.AND.LSL R2, C33, C50, R0
[NwsD---] BIT.ROT.LSR R9, C79, R0
[NswD---] BIT.ROT.LSL R9, C50, R9
[NwsD---] INT.ADD R2.s32, R2.s32, R9.s32
[NswD---] BIT.ROT.LSR R3, C79, R0
[NwsD---] BIT.ROT.LSL R3, C45, R3
[NswD---] BIT.AND.LSR R9, C50, C78, R0
[NwsD---] INT.ADD R3.s32, R3.s32, R9.s32
[NswD---] BIT.AND.LSR R4, C80, SH40, R0
[NwsD---] BIT.ROT.LSL R4, C50, R4
[NswD---] BIT.LUT.AND R9, C59, R0
[NwsD---] INT.ADD R4.s32, R4.s32, R9.s32
[NswD---] INT.ADD R10.s32, R1.s32, R5.s32
[NwsD---] BIT.ROT.LSL R8, C46, R10
[NswD---] INT.ADD R8.s32, R8.s32, R2.s32
[NwsD---] BIT.ROT.LSL R8, C33, R8
[NswD---] FOP.MOV R28, R8
[NwsD---] FOP.MOV R15, R28
[NswD---] INT.ADD R16.s32, R28.s32, SH50.s32
[NwsD---] INT.ADD R17.s32, R28.s32, SH39.s32
[NswD---] INT.ADD R18.s32, R28.s32, SH51.s32
[NwsD---] INT.ADD R30.s32, R28.s32, SH41.s32
[NswD---] INT.ADD R31.s32, R28.s32, SH52.s32
[NwsD---] INT.ADD R9.s32, R28.s32, SH42.s32
[NswD---] INT.ADD R10.s32, R28.s32, SH53.s32
[NwsD---] INT.ADD R144.s32, R28.s32, SH44.s32
[NswD---] INT.ADD R145.s32, R28.s32, SH54.s32
[NwsD---] INT.ADD R146.s32, R28.s32, SH45.s32
[NswD---] INT.ADD R147.s32, R28.s32, SH55.s32
[NwsD---] INT.ADD R148.s32, R28.s32, SH46.s32
[NswD---] INT.ADD R149.s32, R28.s32, SH56.s32
[NwsD---] INT.ADD R150.s32, R28.s32, SH47.s32
[NswD---] INT.ADD R151.s32, R28.s32, SH57.s32
[NwwDs--] DMA.LD.B32 R32, SH[4:5], R15, _, _, _, _, SH[4:6]
[N--Ds--] DMA.LD.B32 R33, SH[4:5], R16, _, _, _, _, SH[4:6]
[N--Ds--] DMA.LD.B32 R34, SH[4:5], R17, _, _, _, _, SH[4:6]
[N--Ds--] DMA.LD.B32 R35, SH[4:5], R18, _, _, _, _, SH[4:6]
[N--Ds--] DMA.LD.B32 R36, SH[4:5], R30, _, _, _, _, SH[4:6]
[N--Ds--] DMA.LD.B32 R37, SH[4:5], R31, _, _, _, _, SH[4:6]
[N--Ds--] DMA.LD.B32 R38, SH[4:5], R9, _, _, _, _, SH[4:6]
[N--Ds--] DMA.LD.B32 R39, SH[4:5], R10, _, _, _, _, SH[4:6]
[N--Ds--] DMA.LD.B32 R40, SH[4:5], R144, _, _, _, _, SH[4:6]
[N--Ds--] DMA.LD.B32 R41, SH[4:5], R145, _, _, _, _, SH[4:6]
[N--Ds--] DMA.LD.B32 R42, SH[4:5], R146, _, _, _, _, SH[4:6]
[N--Ds--] DMA.LD.B32 R43, SH[4:5], R147, _, _, _, _, SH[4:6]
[N--Ds--] DMA.LD.B32 R44, SH[4:5], R148, _, _, _, _, SH[4:6]
[N--Ds--] DMA.LD.B32 R45, SH[4:5], R149, _, _, _, _, SH[4:6]
[N--Ds--] DMA.LD.B32 R46, SH[4:5], R150, _, _, _, _, SH[4:6]
[N--Ds--] DMA.LD.B32 R47, SH[4:5], R151, _, _, _, _, SH[4:6]
[NwwDwww] CTRL.NOP
[NwsD---] INT.ADD R15.s32, R15.s32, SH48.s32
[NswD---] INT.ADD R16.s32, R16.s32, SH48.s32
[NwsD---] INT.ADD R17.s32, R17.s32, SH48.s32
[NswD---] INT.ADD R18.s32, R18.s32, SH48.s32
[NwsD---] INT.ADD R30.s32, R30.s32, SH48.s32
[NswD---] INT.ADD R31.s32, R31.s32, SH48.s32
[NwsD---] INT.ADD R9.s32, R9.s32, SH48.s32
[NswD---] INT.ADD R10.s32, R10.s32, SH48.s32
[NwsD---] INT.ADD R144.s32, R144.s32, SH48.s32
[NswD---] INT.ADD R145.s32, R145.s32, SH48.s32
[NwsD---] INT.ADD R146.s32, R146.s32, SH48.s32
[NswD---] INT.ADD R147.s32, R147.s32, SH48.s32
[NwsD---] INT.ADD R148.s32, R148.s32, SH48.s32
[NswD---] INT.ADD R149.s32, R149.s32, SH48.s32
[NwsD---] INT.ADD R150.s32, R150.s32, SH48.s32
[NswD---] INT.ADD R151.s32, R151.s32, SH48.s32
[NwwDs--] DMA.LD.B32 R48, SH[4:5], R15, _, _, _, _, SH[4:6]
[N--Ds--] DMA.LD.B32 R49, SH[4:5], R16, _, _, _, _, SH[4:6]
[N--Ds--] DMA.LD.B32 R50, SH[4:5], R17, _, _, _, _, SH[4:6]
[N--Ds--] DMA.LD.B32 R51, SH[4:5], R18, _, _, _, _, SH[4:6]
[N--Ds--] DMA.LD.B32 R52, SH[4:5], R30, _, _, _, _, SH[4:6]
[N--Ds--] DMA.LD.B32 R53, SH[4:5], R31, _, _, _, _, SH[4:6]
[N--Ds--] DMA.LD.B32 R54, SH[4:5], R9, _, _, _, _, SH[4:6]
[N--Ds--] DMA.LD.B32 R55, SH[4:5], R10, _, _, _, _, SH[4:6]
[N--Ds--] DMA.LD.B32 R56, SH[4:5], R144, _, _, _, _, SH[4:6]
[N--Ds--] DMA.LD.B32 R57, SH[4:5], R145, _, _, _, _, SH[4:6]
[N--Ds--] DMA.LD.B32 R58, SH[4:5], R146, _, _, _, _, SH[4:6]
[N--Ds--] DMA.LD.B32 R59, SH[4:5], R147, _, _, _, _, SH[4:6]
[N--Ds--] DMA.LD.B32 R60, SH[4:5], R148, _, _, _, _, SH[4:6]
[N--Ds--] DMA.LD.B32 R61, SH[4:5], R149, _, _, _, _, SH[4:6]
[N--Ds--] DMA.LD.B32 R62, SH[4:5], R150, _, _, _, _, SH[4:6]
[N--Ds--] DMA.LD.B32 R63, SH[4:5], R151, _, _, _, _, SH[4:6]
[NwwDwww] CTRL.NOP
[NwsD---] FOP.MOV R64, C0
[NswD---] FOP.MOV R65, C0
[NwsD---] FOP.MOV R66, C0
[NswD---] FOP.MOV R67, C0
[NwsD---] FOP.MOV R68, C0
[NswD---] FOP.MOV R69, C0
[NwsD---] FOP.MOV R70, C0
[NswD---] FOP.MOV R71, C0
[NwsD---] FOP.MOV R72, C0
[NswD---] FOP.MOV R73, C0
[NwsD---] FOP.MOV R74, C0
[NswD---] FOP.MOV R75, C0
[NwsD---] FOP.MOV R76, C0
[NswD---] FOP.MOV R77, C0
[NwsD---] FOP.MOV R78, C0
[NswD---] FOP.MOV R79, C0
[NwsD---] FOP.MOV R144, SH60
[NswD---] FOP.MOV R148, C0
[NwsD---] FOP.MOV R145, SH60
[NswD---] FOP.MOV R149, C0
[NwsD---] FOP.MOV R146, SH60
[NswD---] FOP.MOV R150, C0
[NwsD---] FOP.MOV R147, SH60
[NswD---] FOP.MOV R151, C0
[NwsD---] FOP.MOV R7, C0
[NswD---] INT.BYP.TLE p0, R26.s32, C0.s32
[NwwD---] CTRL.BR.ALLP0T OUTPUT
[NwsD---] INT.BYP.TL p0, R26.s32, SH39.s32
[NwwD---] CTRL.BR.ALLP0T KV_TAIL
KV_LOOP:
[NswD---] INT.ADD R10.s32, R1.s32, R7.s32
[NwsD---] INT.ADD R8.s32, R10.s32, R27.s32
[NswD---] BIT.ROT.LSL R8, C46, R8
[NwsD---] INT.ADD R8.s32, R8.s32, R2.s32
[NswD---] BIT.ROT.LSL R8, C33, R8
[NwsD---] FOP.MOV R28, R8
[NswD---] INT.ADD R15.s32, R28.s32, C0.s32
[NwsD---] INT.ADD R16.s32, R28.s32, SH50.s32
[NwwDs--] DMA.LD.B128 R[152:155], SH[8:9], R15, 4, 4, 32, _, SH[8:10]
[N--Ds--] DMA.LD.B128 R[156:159], SH[8:9], R16, 4, 4, 32, _, SH[8:10]
[NwwDwww] CTRL.NOP
[Ns-D---] MMA.323216 R128[4:2][2:1].f32, R32[2:1][1:1].bf16.row, R152[1:1][2:4].bf16.col, 0
[N-sD---] MMA.323216 R136[4:2][2:1].f32, R34[2:1][1:1].bf16.row, R153[1:1][2:4].bf16.col, 0
[Ns-D---] MMA.323216 R128[4:2][2:1].f32, R36[2:1][1:1].bf16.row, R154[1:1][2:4].bf16.col, R128[4:2][2:1].f32
[N-sD---] MMA.323216 R136[4:2][2:1].f32, R38[2:1][1:1].bf16.row, R155[1:1][2:4].bf16.col, R136[4:2][2:1].f32
[NwwD---] CTRL.NOP
[NswD---] INT.ADD R15.s32, R28.s32, SH44.s32
[NwsD---] INT.ADD R16.s32, R28.s32, SH54.s32
[NwwDs--] DMA.LD.B128 R[152:155], SH[8:9], R15, 4, 4, 32, _, SH[8:10]
[N--Ds--] DMA.LD.B128 R[156:159], SH[8:9], R16, 4, 4, 32, _, SH[8:10]
[NwwDwww] CTRL.NOP
[Ns-D---] MMA.323216 R128[4:2][2:1].f32, R40[2:1][1:1].bf16.row, R152[1:1][2:4].bf16.col, R128[4:2][2:1].f32
[N-sD---] MMA.323216 R136[4:2][2:1].f32, R42[2:1][1:1].bf16.row, R153[1:1][2:4].bf16.col, R136[4:2][2:1].f32
[Ns-D---] MMA.323216 R128[4:2][2:1].f32, R44[2:1][1:1].bf16.row, R154[1:1][2:4].bf16.col, R128[4:2][2:1].f32
[N-sD---] MMA.323216 R136[4:2][2:1].f32, R46[2:1][1:1].bf16.row, R155[1:1][2:4].bf16.col, R136[4:2][2:1].f32
[NwwD---] CTRL.NOP
[NswD---] INT.ADD R15.s32, R28.s32, SH48.s32
[NwsD---] INT.ADD R16.s32, R28.s32, SH58.s32
[NwwDs--] DMA.LD.B128 R[152:155], SH[8:9], R15, 4, 4, 32, _, SH[8:10]
[N--Ds--] DMA.LD.B128 R[156:159], SH[8:9], R16, 4, 4, 32, _, SH[8:10]
[NwwDwww] CTRL.NOP
[Ns-D---] MMA.323216 R128[4:2][2:1].f32, R48[2:1][1:1].bf16.row, R152[1:1][2:4].bf16.col, R128[4:2][2:1].f32
[N-sD---] MMA.323216 R136[4:2][2:1].f32, R50[2:1][1:1].bf16.row, R153[1:1][2:4].bf16.col, R136[4:2][2:1].f32
[Ns-D---] MMA.323216 R128[4:2][2:1].f32, R52[2:1][1:1].bf16.row, R154[1:1][2:4].bf16.col, R128[4:2][2:1].f32
[N-sD---] MMA.323216 R136[4:2][2:1].f32, R54[2:1][1:1].bf16.row, R155[1:1][2:4].bf16.col, R136[4:2][2:1].f32
[NwwD---] CTRL.NOP
[NswD---] INT.ADD R15.s32, R28.s32, SH49.s32
[NwsD---] INT.ADD R16.s32, R28.s32, SH59.s32
[NwwDs--] DMA.LD.B128 R[152:155], SH[8:9], R15, 4, 4, 32, _, SH[8:10]
[N--Ds--] DMA.LD.B128 R[156:159], SH[8:9], R16, 4, 4, 32, _, SH[8:10]
[NwwDwww] CTRL.NOP
[Ns-D---] MMA.323216 R128[4:2][2:1].f32, R56[2:1][1:1].bf16.row, R152[1:1][2:4].bf16.col, R128[4:2][2:1].f32
[N-sD---] MMA.323216 R136[4:2][2:1].f32, R58[2:1][1:1].bf16.row, R153[1:1][2:4].bf16.col, R136[4:2][2:1].f32
[Ns-D---] MMA.323216 R128[4:2][2:1].f32, R60[2:1][1:1].bf16.row, R154[1:1][2:4].bf16.col, R128[4:2][2:1].f32
[N-sD---] MMA.323216 R136[4:2][2:1].f32, R62[2:1][1:1].bf16.row, R155[1:1][2:4].bf16.col, R136[4:2][2:1].f32
[NwwD---] CTRL.NOP
[NswD---] FOP.ADD R128, R128, R136
[NwsD---] FOP.ADD R129, R129, R137
[NswD---] FOP.ADD R130, R130, R138
[NwsD---] FOP.ADD R131, R131, R139
[NswD---] FOP.ADD R132, R132, R140
[NwsD---] FOP.ADD R133, R133, R141
[NswD---] FOP.ADD R134, R134, R142
[NwsD---] FOP.ADD R135, R135, R143
[NswD---] FOP.MUL R128, SH28, R128
[NwsD---] FOP.MUL R129, SH28, R129
[NswD---] FOP.MUL R130, SH28, R130
[NwsD---] FOP.MUL R131, SH28, R131
[NswD---] FOP.MUL R132, SH28, R132
[NwsD---] FOP.MUL R133, SH28, R133
[NswD---] FOP.MUL R134, SH28, R134
[NwsD---] FOP.MUL R135, SH28, R135
[NswD---] FOP.CMP_MOVC.MAX.TST R30, R128, R129
[NwsD---] SHUFFLE.IDX R31, R30, R19, SH43
[NswD---] FOP.CMP_MOVC.MAX.TST R30, R30, R31
[NwsD---] SHUFFLE.IDX R31, R30, R20, SH43
[NswD---] FOP.CMP_MOVC.MAX.TST R30, R30, R31
[NwsD---] SHUFFLE.IDX R31, R30, R21, SH43
[NswD---] FOP.CMP_MOVC.MAX.TST R30, R30, R31
[NwsD---] SHUFFLE.IDX R31, R30, R22, SH43
[NswD---] FOP.CMP_MOVC.MAX.TST R30, R30, R31
[NwsD---] FOP.CMP_MOVC.MAX.TST R30, R30, R144
[NswD---] FOP.ADD R29, R144, R30.neg
[NwsD---] FOP.MOV R144, R30
[NswD---] FOP.ADD R128, R128, R30.neg
[NwsD---] FOP.ADD R129, R129, R30.neg
[NwwD--s] AF32.EXP2 R29, R29
[N--D--s] AF32.EXP2 R128, R128
[N--D--s] AF32.EXP2 R129, R129
[NwwDwww] CTRL.NOP
[NswD---] FOP.ADD R30, R128, R129
[NwsD---] SHUFFLE.IDX R31, R30, R19, SH43
[NswD---] FOP.ADD R30, R30, R31
[NwsD---] SHUFFLE.IDX R31, R30, R20, SH43
[NswD---] FOP.ADD R30, R30, R31
[NwsD---] SHUFFLE.IDX R31, R30, R21, SH43
[NswD---] FOP.ADD R30, R30, R31
[NwsD---] SHUFFLE.IDX R31, R30, R22, SH43
[NswD---] FOP.ADD R30, R30, R31
[NwsD---] FOP.FMA R148, R148, R29, R30
[N--D---] FOP.MUL R64, R64, R29
[N--D---] FOP.MUL R65, R65, R29
[N--D---] FOP.MUL R72, R72, R29
[Ns-D---] FOP.MUL R73, R73, R29
[NswD---] INT.ADD R10.s32, R3.s32, C0.s32
[NwsD---] INT.MADD R13.s32, SH39.s32, R4.s32, R10.s32
[NswD---] INT.ADD R11.s32, R4.s32, C39.s32
[NwsD---] INT.MADD R14.s32, SH39.s32, R11.s32, R10.s32
[NwwD---] CTRL.NOP
[NwwD-s-] LDMA.WR.B32.BST1 R128, R13, X4, 128, BURST_STRIDE=4
[N--D-s-] LDMA.WR.B32.BST1 R129, R14, X4, 128, BURST_STRIDE=4
[NswD---] FOP.CMP_MOVC.MAX.TST R30, R130, R131
[NwsD---] SHUFFLE.IDX R31, R30, R19, SH43
[NswD---] FOP.CMP_MOVC.MAX.TST R30, R30, R31
[NwsD---] SHUFFLE.IDX R31, R30, R20, SH43
[NswD---] FOP.CMP_MOVC.MAX.TST R30, R30, R31
[NwsD---] SHUFFLE.IDX R31, R30, R21, SH43
[NswD---] FOP.CMP_MOVC.MAX.TST R30, R30, R31
[NwsD---] SHUFFLE.IDX R31, R30, R22, SH43
[NswD---] FOP.CMP_MOVC.MAX.TST R30, R30, R31
[NwsD---] FOP.CMP_MOVC.MAX.TST R30, R30, R145
[NswD---] FOP.ADD R29, R145, R30.neg
[NwsD---] FOP.MOV R145, R30
[NswD---] FOP.ADD R130, R130, R30.neg
[NwsD---] FOP.ADD R131, R131, R30.neg
[NwwD--s] AF32.EXP2 R29, R29
[N--D--s] AF32.EXP2 R130, R130
[N--D--s] AF32.EXP2 R131, R131
[NwwDwww] CTRL.NOP
[NswD---] FOP.ADD R30, R130, R131
[NwsD---] SHUFFLE.IDX R31, R30, R19, SH43
[NswD---] FOP.ADD R30, R30, R31
[NwsD---] SHUFFLE.IDX R31, R30, R20, SH43
[NswD---] FOP.ADD R30, R30, R31
[NwsD---] SHUFFLE.IDX R31, R30, R21, SH43
[NswD---] FOP.ADD R30, R30, R31
[NwsD---] SHUFFLE.IDX R31, R30, R22, SH43
[NswD---] FOP.ADD R30, R30, R31
[NwsD---] FOP.FMA R149, R149, R29, R30
[N--D---] FOP.MUL R66, R66, R29
[N--D---] FOP.MUL R67, R67, R29
[N--D---] FOP.MUL R74, R74, R29
[Ns-D---] FOP.MUL R75, R75, R29
[NswD---] INT.ADD R10.s32, R3.s32, C46.s32
[NwsD---] INT.MADD R13.s32, SH39.s32, R4.s32, R10.s32
[NswD---] INT.ADD R11.s32, R4.s32, C39.s32
[NwsD---] INT.MADD R14.s32, SH39.s32, R11.s32, R10.s32
[NwwD---] CTRL.NOP
[NwwD-s-] LDMA.WR.B32.BST1 R130, R13, X4, 128, BURST_STRIDE=4
[N--D-s-] LDMA.WR.B32.BST1 R131, R14, X4, 128, BURST_STRIDE=4
[NswD---] FOP.CMP_MOVC.MAX.TST R30, R132, R133
[NwsD---] SHUFFLE.IDX R31, R30, R19, SH43
[NswD---] FOP.CMP_MOVC.MAX.TST R30, R30, R31
[NwsD---] SHUFFLE.IDX R31, R30, R20, SH43
[NswD---] FOP.CMP_MOVC.MAX.TST R30, R30, R31
[NwsD---] SHUFFLE.IDX R31, R30, R21, SH43
[NswD---] FOP.CMP_MOVC.MAX.TST R30, R30, R31
[NwsD---] SHUFFLE.IDX R31, R30, R22, SH43
[NswD---] FOP.CMP_MOVC.MAX.TST R30, R30, R31
[NwsD---] FOP.CMP_MOVC.MAX.TST R30, R30, R146
[NswD---] FOP.ADD R29, R146, R30.neg
[NwsD---] FOP.MOV R146, R30
[NswD---] FOP.ADD R132, R132, R30.neg
[NwsD---] FOP.ADD R133, R133, R30.neg
[NwwD--s] AF32.EXP2 R29, R29
[N--D--s] AF32.EXP2 R132, R132
[N--D--s] AF32.EXP2 R133, R133
[NwwDwww] CTRL.NOP
[NswD---] FOP.ADD R30, R132, R133
[NwsD---] SHUFFLE.IDX R31, R30, R19, SH43
[NswD---] FOP.ADD R30, R30, R31
[NwsD---] SHUFFLE.IDX R31, R30, R20, SH43
[NswD---] FOP.ADD R30, R30, R31
[NwsD---] SHUFFLE.IDX R31, R30, R21, SH43
[NswD---] FOP.ADD R30, R30, R31
[NwsD---] SHUFFLE.IDX R31, R30, R22, SH43
[NswD---] FOP.ADD R30, R30, R31
[NwsD---] FOP.FMA R150, R150, R29, R30
[N--D---] FOP.MUL R68, R68, R29
[N--D---] FOP.MUL R69, R69, R29
[N--D---] FOP.MUL R76, R76, R29
[Ns-D---] FOP.MUL R77, R77, R29
[NswD---] INT.ADD R10.s32, R3.s32, C39.s32
[NwsD---] INT.MADD R13.s32, SH39.s32, R4.s32, R10.s32
[NswD---] INT.ADD R11.s32, R4.s32, C39.s32
[NwsD---] INT.MADD R14.s32, SH39.s32, R11.s32, R10.s32
[NwwD---] CTRL.NOP
[NwwD-s-] LDMA.WR.B32.BST1 R132, R13, X4, 128, BURST_STRIDE=4
[N--D-s-] LDMA.WR.B32.BST1 R133, R14, X4, 128, BURST_STRIDE=4
[NswD---] FOP.CMP_MOVC.MAX.TST R30, R134, R135
[NwsD---] SHUFFLE.IDX R31, R30, R19, SH43
[NswD---] FOP.CMP_MOVC.MAX.TST R30, R30, R31
[NwsD---] SHUFFLE.IDX R31, R30, R20, SH43
[NswD---] FOP.CMP_MOVC.MAX.TST R30, R30, R31
[NwsD---] SHUFFLE.IDX R31, R30, R21, SH43
[NswD---] FOP.CMP_MOVC.MAX.TST R30, R30, R31
[NwsD---] SHUFFLE.IDX R31, R30, R22, SH43
[NswD---] FOP.CMP_MOVC.MAX.TST R30, R30, R31
[NwsD---] FOP.CMP_MOVC.MAX.TST R30, R30, R147
[NswD---] FOP.ADD R29, R147, R30.neg
[NwsD---] FOP.MOV R147, R30
[NswD---] FOP.ADD R134, R134, R30.neg
[NwsD---] FOP.ADD R135, R135, R30.neg
[NwwD--s] AF32.EXP2 R29, R29
[N--D--s] AF32.EXP2 R134, R134
[N--D--s] AF32.EXP2 R135, R135
[NwwDwww] CTRL.NOP
[NswD---] FOP.ADD R30, R134, R135
[NwsD---] SHUFFLE.IDX R31, R30, R19, SH43
[NswD---] FOP.ADD R30, R30, R31
[NwsD---] SHUFFLE.IDX R31, R30, R20, SH43
[NswD---] FOP.ADD R30, R30, R31
[NwsD---] SHUFFLE.IDX R31, R30, R21, SH43
[NswD---] FOP.ADD R30, R30, R31
[NwsD---] SHUFFLE.IDX R31, R30, R22, SH43
[NswD---] FOP.ADD R30, R30, R31
[NwsD---] FOP.FMA R151, R151, R29, R30
[N--D---] FOP.MUL R70, R70, R29
[N--D---] FOP.MUL R71, R71, R29
[N--D---] FOP.MUL R78, R78, R29
[Ns-D---] FOP.MUL R79, R79, R29
[NswD---] INT.ADD R10.s32, R3.s32, C48.s32
[NwsD---] INT.MADD R13.s32, SH39.s32, R4.s32, R10.s32
[NswD---] INT.ADD R11.s32, R4.s32, C39.s32
[NwsD---] INT.MADD R14.s32, SH39.s32, R11.s32, R10.s32
[NwwD---] CTRL.NOP
[NwwD-s-] LDMA.WR.B32.BST1 R134, R13, X4, 128, BURST_STRIDE=4
[N--D-s-] LDMA.WR.B32.BST1 R135, R14, X4, 128, BURST_STRIDE=4
[NwwDwww] CTRL.NOP
[NswD---] AP.BARRIER LM0
[NwwDwww] CTRL.NOP
[NswD---] INT.ADD R11.s32, R2.s32, C0.s32
[NwsD---] INT.ADD R10.s32, R1.s32, C0.s32
[NswD---] INT.MADD R23.s32, SH39.s32, R11.s32, R10.s32
[NwsD---] INT.ADD R10.s32, R1.s32, C39.s32
[NswD---] INT.MADD R24.s32, SH39.s32, R11.s32, R10.s32
[NwwD-s-] LDMA.RD.B64.BST1 R[30:31], R23, X4, 128, BURST_STRIDE=8
[NwwDwww] CTRL.NOP
[NwsD---] FOP.ADD R156.l.bf16, R30, C0
[NswD---] FOP.ADD R156.h.bf16, R31, C0
[NwwDwww] CTRL.NOP
[N--D-s-] LDMA.RD.B64.BST1 R[30:31], R24, X4, 128, BURST_STRIDE=8
[NwwDwww] CTRL.NOP
[NwsD---] FOP.ADD R157.l.bf16, R30, C0
[NswD---] FOP.ADD R157.h.bf16, R31, C0
[NwsD---] BIT.ROT.LSR R10, C50, R0
[NswD---] INT.ADD R10.s32, R10.s32, C0.s32
[NwsD---] INT.ADD R10.s32, R10.s32, R7.s32
[NswD---] BIT.AND.LSL R11, C33, C59, R0
[NwsD---] INT.ADD R11.s32, R11.s32, R12.s32
[NswD---] INT.ADD R8.s32, R10.s32, R27.s32
[NwsD---] BIT.ROT.LSL R8, C46, R8
[NswD---] INT.ADD R8.s32, R8.s32, R11.s32
[NwsD---] BIT.ROT.LSL R8, C33, R8
[NswD---] FOP.MOV R28, R8
[NwwDs--] DMA.LD.B128 R[152:155], SH[12:13], R28, 4, 4, 32, _, SH[12:14]
[NwwDwww] CTRL.NOP
[Ns-D---] MMA.323216 R64[4:2][2:1].f32, R156[2:1][1:1].bf16.row, R152[1:1][2:1].bf16.row, R64[4:2][2:1].f32
[N-sD---] MMA.323216 R72[4:2][2:1].f32, R156[2:1][1:1].bf16.row, R154[1:1][2:1].bf16.row, R72[4:2][2:1].f32
[NwwDwww] CTRL.NOP
[NwsD---] INT.ADD R11.s32, R2.s32, C39.s32
[NswD---] INT.ADD R10.s32, R1.s32, C0.s32
[NwsD---] INT.MADD R23.s32, SH39.s32, R11.s32, R10.s32
[NswD---] INT.ADD R10.s32, R1.s32, C39.s32
[NwsD---] INT.MADD R24.s32, SH39.s32, R11.s32, R10.s32
[NwwD-s-] LDMA.RD.B64.BST1 R[30:31], R23, X4, 128, BURST_STRIDE=8
[NwwDwww] CTRL.NOP
[NswD---] FOP.ADD R156.l.bf16, R30, C0
[NwsD---] FOP.ADD R156.h.bf16, R31, C0
[NwwDwww] CTRL.NOP
[N--D-s-] LDMA.RD.B64.BST1 R[30:31], R24, X4, 128, BURST_STRIDE=8
[NwwDwww] CTRL.NOP
[NswD---] FOP.ADD R157.l.bf16, R30, C0
[NwsD---] FOP.ADD R157.h.bf16, R31, C0
[NswD---] BIT.ROT.LSR R10, C50, R0
[NwsD---] INT.ADD R10.s32, R10.s32, C39.s32
[NswD---] INT.ADD R10.s32, R10.s32, R7.s32
[NwsD---] BIT.AND.LSL R11, C33, C59, R0
[NswD---] INT.ADD R11.s32, R11.s32, R12.s32
[NwsD---] INT.ADD R8.s32, R10.s32, R27.s32
[NswD---] BIT.ROT.LSL R8, C46, R8
[NwsD---] INT.ADD R8.s32, R8.s32, R11.s32
[NswD---] BIT.ROT.LSL R8, C33, R8
[NwsD---] FOP.MOV R28, R8
[NwwDs--] DMA.LD.B128 R[152:155], SH[12:13], R28, 4, 4, 32, _, SH[12:14]
[NwwDwww] CTRL.NOP
[Ns-D---] MMA.323216 R64[4:2][2:1].f32, R156[2:1][1:1].bf16.row, R152[1:1][2:1].bf16.row, R64[4:2][2:1].f32
[N-sD---] MMA.323216 R72[4:2][2:1].f32, R156[2:1][1:1].bf16.row, R154[1:1][2:1].bf16.row, R72[4:2][2:1].f32
[NwwDwww] CTRL.NOP
[NswD---] AP.BARRIER LM0
[NwwDwww] CTRL.NOP
[NswD---] INT.ADD R7.s32, R7.s32, SH39.s32
[NwsD---] INT.ADD R9.s32, R7.s32, SH39.s32
[NswD---] INT.BYP.TLE p0, R9.s32, R26.s32
[NwwD---] CTRL.BR.ALLP0T KV_LOOP
[NwsD---] INT.BYP.TGE p0, R7.s32, R26.s32
[NwwD---] CTRL.BR.ALLP0T OUTPUT
KV_TAIL:
[NswD---] INT.ADD R10.s32, R1.s32, R7.s32
[NwsD---] INT.ADD R8.s32, R10.s32, R27.s32
[NswD---] BIT.ROT.LSL R8, C46, R8
[NwsD---] INT.ADD R8.s32, R8.s32, R2.s32
[NswD---] BIT.ROT.LSL R8, C33, R8
[NwsD---] FOP.MOV R28, R8
[NswD---] INT.ADD R15.s32, R28.s32, C0.s32
[NwsD---] INT.ADD R16.s32, R28.s32, SH50.s32
[NwwDs--] DMA.LD.B128 R[152:155], SH[8:9], R15, 4, 4, 32, _, SH[8:10]
[N--Ds--] DMA.LD.B128 R[156:159], SH[8:9], R16, 4, 4, 32, _, SH[8:10]
[NwwDwww] CTRL.NOP
[Ns-D---] MMA.323216 R128[4:2][2:1].f32, R32[2:1][1:1].bf16.row, R152[1:1][2:4].bf16.col, 0
[N-sD---] MMA.323216 R136[4:2][2:1].f32, R34[2:1][1:1].bf16.row, R153[1:1][2:4].bf16.col, 0
[Ns-D---] MMA.323216 R128[4:2][2:1].f32, R36[2:1][1:1].bf16.row, R154[1:1][2:4].bf16.col, R128[4:2][2:1].f32
[N-sD---] MMA.323216 R136[4:2][2:1].f32, R38[2:1][1:1].bf16.row, R155[1:1][2:4].bf16.col, R136[4:2][2:1].f32
[NwwD---] CTRL.NOP
[NswD---] INT.ADD R15.s32, R28.s32, SH44.s32
[NwsD---] INT.ADD R16.s32, R28.s32, SH54.s32
[NwwDs--] DMA.LD.B128 R[152:155], SH[8:9], R15, 4, 4, 32, _, SH[8:10]
[N--Ds--] DMA.LD.B128 R[156:159], SH[8:9], R16, 4, 4, 32, _, SH[8:10]
[NwwDwww] CTRL.NOP
[Ns-D---] MMA.323216 R128[4:2][2:1].f32, R40[2:1][1:1].bf16.row, R152[1:1][2:4].bf16.col, R128[4:2][2:1].f32
[N-sD---] MMA.323216 R136[4:2][2:1].f32, R42[2:1][1:1].bf16.row, R153[1:1][2:4].bf16.col, R136[4:2][2:1].f32
[Ns-D---] MMA.323216 R128[4:2][2:1].f32, R44[2:1][1:1].bf16.row, R154[1:1][2:4].bf16.col, R128[4:2][2:1].f32
[N-sD---] MMA.323216 R136[4:2][2:1].f32, R46[2:1][1:1].bf16.row, R155[1:1][2:4].bf16.col, R136[4:2][2:1].f32
[NwwD---] CTRL.NOP
[NswD---] INT.ADD R15.s32, R28.s32, SH48.s32
[NwsD---] INT.ADD R16.s32, R28.s32, SH58.s32
[NwwDs--] DMA.LD.B128 R[152:155], SH[8:9], R15, 4, 4, 32, _, SH[8:10]
[N--Ds--] DMA.LD.B128 R[156:159], SH[8:9], R16, 4, 4, 32, _, SH[8:10]
[NwwDwww] CTRL.NOP
[Ns-D---] MMA.323216 R128[4:2][2:1].f32, R48[2:1][1:1].bf16.row, R152[1:1][2:4].bf16.col, R128[4:2][2:1].f32
[N-sD---] MMA.323216 R136[4:2][2:1].f32, R50[2:1][1:1].bf16.row, R153[1:1][2:4].bf16.col, R136[4:2][2:1].f32
[Ns-D---] MMA.323216 R128[4:2][2:1].f32, R52[2:1][1:1].bf16.row, R154[1:1][2:4].bf16.col, R128[4:2][2:1].f32
[N-sD---] MMA.323216 R136[4:2][2:1].f32, R54[2:1][1:1].bf16.row, R155[1:1][2:4].bf16.col, R136[4:2][2:1].f32
[NwwD---] CTRL.NOP
[NswD---] INT.ADD R15.s32, R28.s32, SH49.s32
[NwsD---] INT.ADD R16.s32, R28.s32, SH59.s32
[NwwDs--] DMA.LD.B128 R[152:155], SH[8:9], R15, 4, 4, 32, _, SH[8:10]
[N--Ds--] DMA.LD.B128 R[156:159], SH[8:9], R16, 4, 4, 32, _, SH[8:10]
[NwwDwww] CTRL.NOP
[Ns-D---] MMA.323216 R128[4:2][2:1].f32, R56[2:1][1:1].bf16.row, R152[1:1][2:4].bf16.col, R128[4:2][2:1].f32
[N-sD---] MMA.323216 R136[4:2][2:1].f32, R58[2:1][1:1].bf16.row, R153[1:1][2:4].bf16.col, R136[4:2][2:1].f32
[Ns-D---] MMA.323216 R128[4:2][2:1].f32, R60[2:1][1:1].bf16.row, R154[1:1][2:4].bf16.col, R128[4:2][2:1].f32
[N-sD---] MMA.323216 R136[4:2][2:1].f32, R62[2:1][1:1].bf16.row, R155[1:1][2:4].bf16.col, R136[4:2][2:1].f32
[NwwD---] CTRL.NOP
[NswD---] FOP.ADD R128, R128, R136
[NwsD---] FOP.ADD R129, R129, R137
[NswD---] FOP.ADD R130, R130, R138
[NwsD---] FOP.ADD R131, R131, R139
[NswD---] FOP.ADD R132, R132, R140
[NwsD---] FOP.ADD R133, R133, R141
[NswD---] FOP.ADD R134, R134, R142
[NwsD---] FOP.ADD R135, R135, R143
[NswD---] FOP.MUL R128, SH28, R128
[NwsD---] FOP.MUL R129, SH28, R129
[NswD---] FOP.MUL R130, SH28, R130
[NwsD---] FOP.MUL R131, SH28, R131
[NswD---] FOP.MUL R132, SH28, R132
[NwsD---] FOP.MUL R133, SH28, R133
[NswD---] FOP.MUL R134, SH28, R134
[NwsD---] FOP.MUL R135, SH28, R135
[NswD---] INT.ADD R11.s32, R4.s32, C0.s32
[NwsD---] INT.ADD R11.s32, R11.s32, R7.s32
[NswD---] INT.BYP.TGE p0, R11.s32, R26.s32
[NwsD---] P0 FOP.MOV R128, SH60
[NswD---] P0 FOP.MOV R130, SH60
[NwsD---] P0 FOP.MOV R132, SH60
[NswD---] P0 FOP.MOV R134, SH60
[NwsD---] INT.ADD R11.s32, R4.s32, C39.s32
[NswD---] INT.ADD R11.s32, R11.s32, R7.s32
[NwsD---] INT.BYP.TGE p0, R11.s32, R26.s32
[NswD---] P0 FOP.MOV R129, SH60
[NwsD---] P0 FOP.MOV R131, SH60
[NswD---] P0 FOP.MOV R133, SH60
[NwsD---] P0 FOP.MOV R135, SH60
[NswD---] FOP.CMP_MOVC.MAX.TST R30, R128, R129
[NwsD---] SHUFFLE.IDX R31, R30, R19, SH43
[NswD---] FOP.CMP_MOVC.MAX.TST R30, R30, R31
[NwsD---] SHUFFLE.IDX R31, R30, R20, SH43
[NswD---] FOP.CMP_MOVC.MAX.TST R30, R30, R31
[NwsD---] SHUFFLE.IDX R31, R30, R21, SH43
[NswD---] FOP.CMP_MOVC.MAX.TST R30, R30, R31
[NwsD---] SHUFFLE.IDX R31, R30, R22, SH43
[NswD---] FOP.CMP_MOVC.MAX.TST R30, R30, R31
[NwsD---] FOP.CMP_MOVC.MAX.TST R30, R30, R144
[NswD---] FOP.ADD R29, R144, R30.neg
[NwsD---] FOP.MOV R144, R30
[NswD---] FOP.ADD R128, R128, R30.neg
[NwsD---] FOP.ADD R129, R129, R30.neg
[NwwD--s] AF32.EXP2 R29, R29
[N--D--s] AF32.EXP2 R128, R128
[N--D--s] AF32.EXP2 R129, R129
[NwwDwww] CTRL.NOP
[NswD---] FOP.ADD R30, R128, R129
[NwsD---] SHUFFLE.IDX R31, R30, R19, SH43
[NswD---] FOP.ADD R30, R30, R31
[NwsD---] SHUFFLE.IDX R31, R30, R20, SH43
[NswD---] FOP.ADD R30, R30, R31
[NwsD---] SHUFFLE.IDX R31, R30, R21, SH43
[NswD---] FOP.ADD R30, R30, R31
[NwsD---] SHUFFLE.IDX R31, R30, R22, SH43
[NswD---] FOP.ADD R30, R30, R31
[NwsD---] FOP.FMA R148, R148, R29, R30
[N--D---] FOP.MUL R64, R64, R29
[N--D---] FOP.MUL R65, R65, R29
[N--D---] FOP.MUL R72, R72, R29
[Ns-D---] FOP.MUL R73, R73, R29
[NswD---] INT.ADD R10.s32, R3.s32, C0.s32
[NwsD---] INT.MADD R13.s32, SH39.s32, R4.s32, R10.s32
[NswD---] INT.ADD R11.s32, R4.s32, C39.s32
[NwsD---] INT.MADD R14.s32, SH39.s32, R11.s32, R10.s32
[NwwD---] CTRL.NOP
[NwwD-s-] LDMA.WR.B32.BST1 R128, R13, X4, 128, BURST_STRIDE=4
[N--D-s-] LDMA.WR.B32.BST1 R129, R14, X4, 128, BURST_STRIDE=4
[NswD---] FOP.CMP_MOVC.MAX.TST R30, R130, R131
[NwsD---] SHUFFLE.IDX R31, R30, R19, SH43
[NswD---] FOP.CMP_MOVC.MAX.TST R30, R30, R31
[NwsD---] SHUFFLE.IDX R31, R30, R20, SH43
[NswD---] FOP.CMP_MOVC.MAX.TST R30, R30, R31
[NwsD---] SHUFFLE.IDX R31, R30, R21, SH43
[NswD---] FOP.CMP_MOVC.MAX.TST R30, R30, R31
[NwsD---] SHUFFLE.IDX R31, R30, R22, SH43
[NswD---] FOP.CMP_MOVC.MAX.TST R30, R30, R31
[NwsD---] FOP.CMP_MOVC.MAX.TST R30, R30, R145
[NswD---] FOP.ADD R29, R145, R30.neg
[NwsD---] FOP.MOV R145, R30
[NswD---] FOP.ADD R130, R130, R30.neg
[NwsD---] FOP.ADD R131, R131, R30.neg
[NwwD--s] AF32.EXP2 R29, R29
[N--D--s] AF32.EXP2 R130, R130
[N--D--s] AF32.EXP2 R131, R131
[NwwDwww] CTRL.NOP
[NswD---] FOP.ADD R30, R130, R131
[NwsD---] SHUFFLE.IDX R31, R30, R19, SH43
[NswD---] FOP.ADD R30, R30, R31
[NwsD---] SHUFFLE.IDX R31, R30, R20, SH43
[NswD---] FOP.ADD R30, R30, R31
[NwsD---] SHUFFLE.IDX R31, R30, R21, SH43
[NswD---] FOP.ADD R30, R30, R31
[NwsD---] SHUFFLE.IDX R31, R30, R22, SH43
[NswD---] FOP.ADD R30, R30, R31
[NwsD---] FOP.FMA R149, R149, R29, R30
[N--D---] FOP.MUL R66, R66, R29
[N--D---] FOP.MUL R67, R67, R29
[N--D---] FOP.MUL R74, R74, R29
[Ns-D---] FOP.MUL R75, R75, R29
[NswD---] INT.ADD R10.s32, R3.s32, C46.s32
[NwsD---] INT.MADD R13.s32, SH39.s32, R4.s32, R10.s32
[NswD---] INT.ADD R11.s32, R4.s32, C39.s32
[NwsD---] INT.MADD R14.s32, SH39.s32, R11.s32, R10.s32
[NwwD---] CTRL.NOP
[NwwD-s-] LDMA.WR.B32.BST1 R130, R13, X4, 128, BURST_STRIDE=4
[N--D-s-] LDMA.WR.B32.BST1 R131, R14, X4, 128, BURST_STRIDE=4
[NswD---] FOP.CMP_MOVC.MAX.TST R30, R132, R133
[NwsD---] SHUFFLE.IDX R31, R30, R19, SH43
[NswD---] FOP.CMP_MOVC.MAX.TST R30, R30, R31
[NwsD---] SHUFFLE.IDX R31, R30, R20, SH43
[NswD---] FOP.CMP_MOVC.MAX.TST R30, R30, R31
[NwsD---] SHUFFLE.IDX R31, R30, R21, SH43
[NswD---] FOP.CMP_MOVC.MAX.TST R30, R30, R31
[NwsD---] SHUFFLE.IDX R31, R30, R22, SH43
[NswD---] FOP.CMP_MOVC.MAX.TST R30, R30, R31
[NwsD---] FOP.CMP_MOVC.MAX.TST R30, R30, R146
[NswD---] FOP.ADD R29, R146, R30.neg
[NwsD---] FOP.MOV R146, R30
[NswD---] FOP.ADD R132, R132, R30.neg
[NwsD---] FOP.ADD R133, R133, R30.neg
[NwwD--s] AF32.EXP2 R29, R29
[N--D--s] AF32.EXP2 R132, R132
[N--D--s] AF32.EXP2 R133, R133
[NwwDwww] CTRL.NOP
[NswD---] FOP.ADD R30, R132, R133
[NwsD---] SHUFFLE.IDX R31, R30, R19, SH43
[NswD---] FOP.ADD R30, R30, R31
[NwsD---] SHUFFLE.IDX R31, R30, R20, SH43
[NswD---] FOP.ADD R30, R30, R31
[NwsD---] SHUFFLE.IDX R31, R30, R21, SH43
[NswD---] FOP.ADD R30, R30, R31
[NwsD---] SHUFFLE.IDX R31, R30, R22, SH43
[NswD---] FOP.ADD R30, R30, R31
[NwsD---] FOP.FMA R150, R150, R29, R30
[N--D---] FOP.MUL R68, R68, R29
[N--D---] FOP.MUL R69, R69, R29
[N--D---] FOP.MUL R76, R76, R29
[Ns-D---] FOP.MUL R77, R77, R29
[NswD---] INT.ADD R10.s32, R3.s32, C39.s32
[NwsD---] INT.MADD R13.s32, SH39.s32, R4.s32, R10.s32
[NswD---] INT.ADD R11.s32, R4.s32, C39.s32
[NwsD---] INT.MADD R14.s32, SH39.s32, R11.s32, R10.s32
[NwwD---] CTRL.NOP
[NwwD-s-] LDMA.WR.B32.BST1 R132, R13, X4, 128, BURST_STRIDE=4
[N--D-s-] LDMA.WR.B32.BST1 R133, R14, X4, 128, BURST_STRIDE=4
[NswD---] FOP.CMP_MOVC.MAX.TST R30, R134, R135
[NwsD---] SHUFFLE.IDX R31, R30, R19, SH43
[NswD---] FOP.CMP_MOVC.MAX.TST R30, R30, R31
[NwsD---] SHUFFLE.IDX R31, R30, R20, SH43
[NswD---] FOP.CMP_MOVC.MAX.TST R30, R30, R31
[NwsD---] SHUFFLE.IDX R31, R30, R21, SH43
[NswD---] FOP.CMP_MOVC.MAX.TST R30, R30, R31
[NwsD---] SHUFFLE.IDX R31, R30, R22, SH43
[NswD---] FOP.CMP_MOVC.MAX.TST R30, R30, R31
[NwsD---] FOP.CMP_MOVC.MAX.TST R30, R30, R147
[NswD---] FOP.ADD R29, R147, R30.neg
[NwsD---] FOP.MOV R147, R30
[NswD---] FOP.ADD R134, R134, R30.neg
[NwsD---] FOP.ADD R135, R135, R30.neg
[NwwD--s] AF32.EXP2 R29, R29
[N--D--s] AF32.EXP2 R134, R134
[N--D--s] AF32.EXP2 R135, R135
[NwwDwww] CTRL.NOP
[NswD---] FOP.ADD R30, R134, R135
[NwsD---] SHUFFLE.IDX R31, R30, R19, SH43
[NswD---] FOP.ADD R30, R30, R31
[NwsD---] SHUFFLE.IDX R31, R30, R20, SH43
[NswD---] FOP.ADD R30, R30, R31
[NwsD---] SHUFFLE.IDX R31, R30, R21, SH43
[NswD---] FOP.ADD R30, R30, R31
[NwsD---] SHUFFLE.IDX R31, R30, R22, SH43
[NswD---] FOP.ADD R30, R30, R31
[NwsD---] FOP.FMA R151, R151, R29, R30
[N--D---] FOP.MUL R70, R70, R29
[N--D---] FOP.MUL R71, R71, R29
[N--D---] FOP.MUL R78, R78, R29
[Ns-D---] FOP.MUL R79, R79, R29
[NswD---] INT.ADD R10.s32, R3.s32, C48.s32
[NwsD---] INT.MADD R13.s32, SH39.s32, R4.s32, R10.s32
[NswD---] INT.ADD R11.s32, R4.s32, C39.s32
[NwsD---] INT.MADD R14.s32, SH39.s32, R11.s32, R10.s32
[NwwD---] CTRL.NOP
[NwwD-s-] LDMA.WR.B32.BST1 R134, R13, X4, 128, BURST_STRIDE=4
[N--D-s-] LDMA.WR.B32.BST1 R135, R14, X4, 128, BURST_STRIDE=4
[NwwDwww] CTRL.NOP
[NswD---] AP.BARRIER LM0
[NwwDwww] CTRL.NOP
[NswD---] INT.ADD R11.s32, R2.s32, C0.s32
[NwsD---] INT.ADD R10.s32, R1.s32, C0.s32
[NswD---] INT.MADD R23.s32, SH39.s32, R11.s32, R10.s32
[NwsD---] INT.ADD R10.s32, R1.s32, C39.s32
[NswD---] INT.MADD R24.s32, SH39.s32, R11.s32, R10.s32
[NwwD-s-] LDMA.RD.B64.BST1 R[30:31], R23, X4, 128, BURST_STRIDE=8
[NwwDwww] CTRL.NOP
[NwsD---] FOP.ADD R156.l.bf16, R30, C0
[NswD---] FOP.ADD R156.h.bf16, R31, C0
[NwwDwww] CTRL.NOP
[N--D-s-] LDMA.RD.B64.BST1 R[30:31], R24, X4, 128, BURST_STRIDE=8
[NwwDwww] CTRL.NOP
[NwsD---] FOP.ADD R157.l.bf16, R30, C0
[NswD---] FOP.ADD R157.h.bf16, R31, C0
[NwsD---] BIT.ROT.LSR R10, C50, R0
[NswD---] INT.ADD R10.s32, R10.s32, C0.s32
[NwsD---] INT.ADD R10.s32, R10.s32, R7.s32
[NswD---] BIT.AND.LSL R11, C33, C59, R0
[NwsD---] INT.ADD R11.s32, R11.s32, R12.s32
[NswD---] INT.ADD R8.s32, R10.s32, R27.s32
[NwsD---] BIT.ROT.LSL R8, C46, R8
[NswD---] INT.ADD R8.s32, R8.s32, R11.s32
[NwsD---] BIT.ROT.LSL R8, C33, R8
[NswD---] FOP.MOV R28, R8
[NwwDs--] DMA.LD.B128 R[152:155], SH[12:13], R28, 4, 4, 32, _, SH[12:14]
[NwwDwww] CTRL.NOP
[NwsD---] INT.BYP.TGE p0, R10.s32, R26.s32
[NswD---] P0 FOP.MOV R152, C0
[NwsD---] P0 FOP.MOV R153, C0
[NswD---] P0 FOP.MOV R154, C0
[NwsD---] P0 FOP.MOV R155, C0
[Ns-D---] MMA.323216 R64[4:2][2:1].f32, R156[2:1][1:1].bf16.row, R152[1:1][2:1].bf16.row, R64[4:2][2:1].f32
[N-sD---] MMA.323216 R72[4:2][2:1].f32, R156[2:1][1:1].bf16.row, R154[1:1][2:1].bf16.row, R72[4:2][2:1].f32
[NwwDwww] CTRL.NOP
[NswD---] INT.ADD R11.s32, R2.s32, C39.s32
[NwsD---] INT.ADD R10.s32, R1.s32, C0.s32
[NswD---] INT.MADD R23.s32, SH39.s32, R11.s32, R10.s32
[NwsD---] INT.ADD R10.s32, R1.s32, C39.s32
[NswD---] INT.MADD R24.s32, SH39.s32, R11.s32, R10.s32
[NwwD-s-] LDMA.RD.B64.BST1 R[30:31], R23, X4, 128, BURST_STRIDE=8
[NwwDwww] CTRL.NOP
[NwsD---] FOP.ADD R156.l.bf16, R30, C0
[NswD---] FOP.ADD R156.h.bf16, R31, C0
[NwwDwww] CTRL.NOP
[N--D-s-] LDMA.RD.B64.BST1 R[30:31], R24, X4, 128, BURST_STRIDE=8
[NwwDwww] CTRL.NOP
[NwsD---] FOP.ADD R157.l.bf16, R30, C0
[NswD---] FOP.ADD R157.h.bf16, R31, C0
[NwsD---] BIT.ROT.LSR R10, C50, R0
[NswD---] INT.ADD R10.s32, R10.s32, C39.s32
[NwsD---] INT.ADD R10.s32, R10.s32, R7.s32
[NswD---] BIT.AND.LSL R11, C33, C59, R0
[NwsD---] INT.ADD R11.s32, R11.s32, R12.s32
[NswD---] INT.ADD R8.s32, R10.s32, R27.s32
[NwsD---] BIT.ROT.LSL R8, C46, R8
[NswD---] INT.ADD R8.s32, R8.s32, R11.s32
[NwsD---] BIT.ROT.LSL R8, C33, R8
[NswD---] FOP.MOV R28, R8
[NwwDs--] DMA.LD.B128 R[152:155], SH[12:13], R28, 4, 4, 32, _, SH[12:14]
[NwwDwww] CTRL.NOP
[NwsD---] INT.BYP.TGE p0, R10.s32, R26.s32
[NswD---] P0 FOP.MOV R152, C0
[NwsD---] P0 FOP.MOV R153, C0
[NswD---] P0 FOP.MOV R154, C0
[NwsD---] P0 FOP.MOV R155, C0
[Ns-D---] MMA.323216 R64[4:2][2:1].f32, R156[2:1][1:1].bf16.row, R152[1:1][2:1].bf16.row, R64[4:2][2:1].f32
[N-sD---] MMA.323216 R72[4:2][2:1].f32, R156[2:1][1:1].bf16.row, R154[1:1][2:1].bf16.row, R72[4:2][2:1].f32
[NwwDwww] CTRL.NOP
[NswD---] AP.BARRIER LM0
[NwwDwww] CTRL.NOP
OUTPUT:
[NswD---] FOP.CMP_MOVC.MAX.TST R148, R148, C1
[NwsD---] AF32.RCP R148, R148
[NwwDwww] CTRL.NOP
[NswD---] INT.ADD R10.s32, R3.s32, C0.s32
[NwsD---] INT.ADD R10.s32, R10.s32, R5.s32
[NswD---] INT.ADD R11.s32, R4.s32, R12.s32
[NwsD---] BIT.ROT.LSL R8, C46, R10
[NswD---] INT.ADD R8.s32, R8.s32, R11.s32
[NwsD---] BIT.ROT.LSL R8, C33, R8
[NswD---] FOP.MOV R28, R8
[NwsD---] INT.BYP.TL p0, R10.s32, R25.s32
[NswD---] INT.ADD R8.s32, R28.s32, C0.s32
[NwsD---] FOP.MUL R64.l.bf16, R64, R148
[NwwD--s] P0 DMA.ST.B16 R64.l, SH[0:1], R8
[NwwDwww] CTRL.NOP
[NswD---] INT.ADD R8.s32, R28.s32, SH39.s32
[NwsD---] FOP.MUL R65.l.bf16, R65, R148
[NwwD--s] P0 DMA.ST.B16 R65.l, SH[0:1], R8
[NwwDwww] CTRL.NOP
[NswD---] INT.ADD R8.s32, R28.s32, SH41.s32
[NwsD---] FOP.MUL R72.l.bf16, R72, R148
[NwwD--s] P0 DMA.ST.B16 R72.l, SH[0:1], R8
[NwwDwww] CTRL.NOP
[NswD---] INT.ADD R8.s32, R28.s32, SH42.s32
[NwsD---] FOP.MUL R73.l.bf16, R73, R148
[NwwD--s] P0 DMA.ST.B16 R73.l, SH[0:1], R8
[NwwDwww] CTRL.NOP
[NswD---] FOP.CMP_MOVC.MAX.TST R149, R149, C1
[NwsD---] AF32.RCP R149, R149
[NwwDwww] CTRL.NOP
[NswD---] INT.ADD R10.s32, R3.s32, C46.s32
[NwsD---] INT.ADD R10.s32, R10.s32, R5.s32
[NswD---] INT.ADD R11.s32, R4.s32, R12.s32
[NwsD---] BIT.ROT.LSL R8, C46, R10
[NswD---] INT.ADD R8.s32, R8.s32, R11.s32
[NwsD---] BIT.ROT.LSL R8, C33, R8
[NswD---] FOP.MOV R28, R8
[NwsD---] INT.BYP.TL p0, R10.s32, R25.s32
[NswD---] INT.ADD R8.s32, R28.s32, C0.s32
[NwsD---] FOP.MUL R66.l.bf16, R66, R149
[NwwD--s] P0 DMA.ST.B16 R66.l, SH[0:1], R8
[NwwDwww] CTRL.NOP
[NswD---] INT.ADD R8.s32, R28.s32, SH39.s32
[NwsD---] FOP.MUL R67.l.bf16, R67, R149
[NwwD--s] P0 DMA.ST.B16 R67.l, SH[0:1], R8
[NwwDwww] CTRL.NOP
[NswD---] INT.ADD R8.s32, R28.s32, SH41.s32
[NwsD---] FOP.MUL R74.l.bf16, R74, R149
[NwwD--s] P0 DMA.ST.B16 R74.l, SH[0:1], R8
[NwwDwww] CTRL.NOP
[NswD---] INT.ADD R8.s32, R28.s32, SH42.s32
[NwsD---] FOP.MUL R75.l.bf16, R75, R149
[NwwD--s] P0 DMA.ST.B16 R75.l, SH[0:1], R8
[NwwDwww] CTRL.NOP
[NswD---] FOP.CMP_MOVC.MAX.TST R150, R150, C1
[NwsD---] AF32.RCP R150, R150
[NwwDwww] CTRL.NOP
[NswD---] INT.ADD R10.s32, R3.s32, C39.s32
[NwsD---] INT.ADD R10.s32, R10.s32, R5.s32
[NswD---] INT.ADD R11.s32, R4.s32, R12.s32
[NwsD---] BIT.ROT.LSL R8, C46, R10
[NswD---] INT.ADD R8.s32, R8.s32, R11.s32
[NwsD---] BIT.ROT.LSL R8, C33, R8
[NswD---] FOP.MOV R28, R8
[NwsD---] INT.BYP.TL p0, R10.s32, R25.s32
[NswD---] INT.ADD R8.s32, R28.s32, C0.s32
[NwsD---] FOP.MUL R68.l.bf16, R68, R150
[NwwD--s] P0 DMA.ST.B16 R68.l, SH[0:1], R8
[NwwDwww] CTRL.NOP
[NswD---] INT.ADD R8.s32, R28.s32, SH39.s32
[NwsD---] FOP.MUL R69.l.bf16, R69, R150
[NwwD--s] P0 DMA.ST.B16 R69.l, SH[0:1], R8
[NwwDwww] CTRL.NOP
[NswD---] INT.ADD R8.s32, R28.s32, SH41.s32
[NwsD---] FOP.MUL R76.l.bf16, R76, R150
[NwwD--s] P0 DMA.ST.B16 R76.l, SH[0:1], R8
[NwwDwww] CTRL.NOP
[NswD---] INT.ADD R8.s32, R28.s32, SH42.s32
[NwsD---] FOP.MUL R77.l.bf16, R77, R150
[NwwD--s] P0 DMA.ST.B16 R77.l, SH[0:1], R8
[NwwDwww] CTRL.NOP
[NswD---] FOP.CMP_MOVC.MAX.TST R151, R151, C1
[NwsD---] AF32.RCP R151, R151
[NwwDwww] CTRL.NOP
[NswD---] INT.ADD R10.s32, R3.s32, C48.s32
[NwsD---] INT.ADD R10.s32, R10.s32, R5.s32
[NswD---] INT.ADD R11.s32, R4.s32, R12.s32
[NwsD---] BIT.ROT.LSL R8, C46, R10
[NswD---] INT.ADD R8.s32, R8.s32, R11.s32
[NwsD---] BIT.ROT.LSL R8, C33, R8
[NswD---] FOP.MOV R28, R8
[NwsD---] INT.BYP.TL p0, R10.s32, R25.s32
[NswD---] INT.ADD R8.s32, R28.s32, C0.s32
[NwsD---] FOP.MUL R70.l.bf16, R70, R151
[NwwD--s] P0 DMA.ST.B16 R70.l, SH[0:1], R8
[NwwDwww] CTRL.NOP
[NswD---] INT.ADD R8.s32, R28.s32, SH39.s32
[NwsD---] FOP.MUL R71.l.bf16, R71, R151
[NwwD--s] P0 DMA.ST.B16 R71.l, SH[0:1], R8
[NwwDwww] CTRL.NOP
[NswD---] INT.ADD R8.s32, R28.s32, SH41.s32
[NwsD---] FOP.MUL R78.l.bf16, R78, R151
[NwwD--s] P0 DMA.ST.B16 R78.l, SH[0:1], R8
[NwwDwww] CTRL.NOP
[NswD---] INT.ADD R8.s32, R28.s32, SH42.s32
[NwsD---] FOP.MUL R79.l.bf16, R79, R151
[NwwD--s] P0 DMA.ST.B16 R79.l, SH[0:1], R8
[NwwDwww] CTRL.NOP
FINISH:
[NwwD---] CTRL.END

.Lfunc_end0:
    .size   mate_fa_d256_mp22, .Lfunc_end0-mate_fa_d256_mp22




    .type   llvm.mtgpu.kernel.mate_fa_d256_mp22.sm,@object
    .local  llvm.mtgpu.kernel.mate_fa_d256_mp22.sm
    .comm   llvm.mtgpu.kernel.mate_fa_d256_mp22.sm,4096,4

    .addrsig
    .mtgpu_metadata.start
---
mtgpu.kernels:
  - .arch_id:        220
    .args:
      - .argBase:
          .address_space:  0
          .align:          16
          .isVolatile:     false
          .layout:         0
          .name:           out
          .offset:         0
          .reg_bank:       3
          .size:           16
          .type:           25
        .input_mode:     0
        .kind:           18
      - .argBase:
          .address_space:  0
          .align:          16
          .isVolatile:     false
          .layout:         0
          .name:           q
          .offset:         4
          .reg_bank:       3
          .size:           16
          .type:           25
        .input_mode:     0
        .kind:           18
      - .argBase:
          .address_space:  0
          .align:          16
          .isVolatile:     false
          .layout:         0
          .name:           k
          .offset:         8
          .reg_bank:       3
          .size:           16
          .type:           25
        .input_mode:     0
        .kind:           18
      - .argBase:
          .address_space:  0
          .align:          16
          .isVolatile:     false
          .layout:         0
          .name:           v
          .offset:         12
          .reg_bank:       3
          .size:           16
          .type:           25
        .input_mode:     0
        .kind:           18
      - .argBase:
          .address_space:  0
          .align:          16
          .isVolatile:     false
          .layout:         0
          .name:           cuq
          .offset:         16
          .reg_bank:       3
          .size:           16
          .type:           25
        .input_mode:     0
        .kind:           18
      - .argBase:
          .address_space:  0
          .align:          16
          .isVolatile:     false
          .layout:         0
          .name:           usedk
          .offset:         20
          .reg_bank:       3
          .size:           16
          .type:           25
        .input_mode:     0
        .kind:           18
      - .argBase:
          .address_space:  0
          .align:          4
          .isVolatile:     false
          .layout:         0
          .name:           sq
          .offset:         24
          .reg_bank:       3
          .size:           4
          .type:           6
        .input_mode:     0
        .kind:           18
      - .argBase:
          .address_space:  0
          .align:          4
          .isVolatile:     false
          .layout:         0
          .name:           sk
          .offset:         25
          .reg_bank:       3
          .size:           4
          .type:           6
        .input_mode:     0
        .kind:           18
      - .argBase:
          .address_space:  0
          .align:          4
          .isVolatile:     false
          .layout:         0
          .name:           hq
          .offset:         26
          .reg_bank:       3
          .size:           4
          .type:           6
        .input_mode:     0
        .kind:           18
      - .argBase:
          .address_space:  0
          .align:          4
          .isVolatile:     false
          .layout:         0
          .name:           hkv
          .offset:         27
          .reg_bank:       3
          .size:           4
          .type:           6
        .input_mode:     0
        .kind:           18
      - .argBase:
          .address_space:  0
          .align:          4
          .isVolatile:     false
          .layout:         0
          .name:           scale_log2
          .offset:         28
          .reg_bank:       3
          .size:           4
          .type:           8
        .input_mode:     0
        .kind:           18
      - .argBase:
          .address_space:  0
          .align:          0
          .isVolatile:     false
          .layout:         0
          .name:           prog_size
          .offset:         29
          .reg_bank:       3
          .size:           4
          .type:           7
        .input_mode:     0
        .kind:           35
      - .argBase:
          .address_space:  0
          .align:          0
          .isVolatile:     false
          .layout:         0
          .name:           prog_addr
          .offset:         30
          .reg_bank:       3
          .size:           8
          .type:           10
        .input_mode:     0
        .kind:           33
      - .argBase:
          .address_space:  0
          .align:          0
          .isVolatile:     false
          .layout:         0
          .name:           gridDim_x
          .offset:         32
          .reg_bank:       3
          .size:           4
          .type:           0
        .input_mode:     0
        .kind:           13
      - .argBase:
          .address_space:  0
          .align:          0
          .isVolatile:     false
          .layout:         0
          .name:           gridDim_y
          .offset:         33
          .reg_bank:       3
          .size:           4
          .type:           0
        .input_mode:     0
        .kind:           14
      - .argBase:
          .address_space:  0
          .align:          0
          .isVolatile:     false
          .layout:         0
          .name:           gridDim_z
          .offset:         34
          .reg_bank:       3
          .size:           4
          .type:           0
        .input_mode:     0
        .kind:           15
      - .argBase:
          .address_space:  0
          .align:          0
          .isVolatile:     false
          .layout:         0
          .name:           blockDim_x
          .offset:         35
          .reg_bank:       3
          .size:           4
          .type:           0
        .input_mode:     0
        .kind:           9
      - .argBase:
          .address_space:  0
          .align:          0
          .isVolatile:     false
          .layout:         0
          .name:           Global Spill Base
          .offset:         36
          .reg_bank:       3
          .size:           8
          .type:           18
        .input_mode:     0
        .kind:           19
      - .argBase:
          .address_space:  0
          .align:          0
          .isVolatile:     false
          .layout:         0
          .name:           Core Number
          .offset:         38
          .reg_bank:       3
          .size:           4
          .type:           7
        .input_mode:     0
        .kind:           23
      - .argBase:
          .address_space:  0
          .align:          0
          .isVolatile:     false
          .layout:         0
          .name:           blockIdx_x
          .offset:         4
          .reg_bank:       5
          .size:           4
          .type:           0
        .input_mode:     0
        .kind:           5
      - .argBase:
          .address_space:  0
          .align:          0
          .isVolatile:     false
          .layout:         0
          .name:           blockIdx_y
          .offset:         5
          .reg_bank:       5
          .size:           4
          .type:           0
        .input_mode:     0
        .kind:           6
      - .argBase:
          .address_space:  0
          .align:          0
          .isVolatile:     false
          .layout:         0
          .name:           blockIdx_z
          .offset:         6
          .reg_bank:       5
          .size:           4
          .type:           0
        .input_mode:     0
        .kind:           7
      - .argBase:
          .address_space:  0
          .align:          0
          .isVolatile:     false
          .layout:         0
          .name:           threadIdx_x
          .offset:         0
          .reg_bank:       2
          .size:           4
          .type:           0
        .input_mode:     0
        .kind:           1
    .async_bar_count: 0
    .attr_reg_count: 1
    .attrs:          {}
    .barrier_coef:   0
    .clusterX:       0
    .clusterY:       0
    .clusterZ:       0
    .coef_reg_count: 0
    .const_calc_kernel: !str ''
    .constcalc_temp_reg_count: 0
    .global_atomic_count: 0
    .has_barrier:    true
    .has_cluster_barrier: false
    .has_texture: false
    .has_sqmma_conv:    false
    .imm_info:
      .offset:         39
      .reg_bank:       3
      .size:           88
      .values:
        - 32
        - 63
        - 64
        - 96
        - 127
        - 128
        - 160
        - 192
        - 224
        - 256
        - 384
        - 8192
        - 8224
        - 8256
        - 8288
        - 8320
        - 8352
        - 8384
        - 8416
        - 8448
        - 8576
        - 4286578687
    .indirect_call_flag: false
    .call_flag: false
    .internal_reg_count: 0
    .local_mem_used_in_cluster: 0
    .max_block_size: 1
    .name:           mate_fa_d256_mp22
    .private_memory_size: 0
    .shared_memory_size: 4224
    .shared_reg_count: 61
    .slot_reg_count: 0
    .temp_reg_count: 160
    .wave_mode:      0
mtgpu.version:
  - 2
  - 2
...

    .mtgpu_metadata.end
