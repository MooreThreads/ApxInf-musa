
	.text
	.protected	apxinf_qy2_bf16flash_atten_512_256x64x128
	.globl	apxinf_qy2_bf16flash_atten_512_256x64x128
	.p2align	8
	.type	apxinf_qy2_bf16flash_atten_512_256x64x128,@function
apxinf_qy2_bf16flash_atten_512_256x64x128:






[N--D--s]    MOV.LD.LM.B32 R48, LM6
[N--D--s]    MOV.LD.LM.B32 R49, LM5
[N--D--o]    MOV.LD.LM.B32 R21, LM4
[NswD---] INT.ADD R20.s32, SH52.s32, R49.s32
[NwsD---] INT.BYP.TE p0, SH53.s32, C33.s32
[NswD---] !p0 INT.MUL R[40:41].u64, SH54.u32, R20.u32
[NwsD---] !p0 BIT.ROT.LSR R42, SH55, R41
[NswD---] INT.BYP.MOVC.PRB R42.s32, R20.s32, R42.s32, p0
[NwsD---]    BIT.ROT.LSR SL3, C59, AR0
[NswD---]    BIT.LUT.AND R0, SH81, AR0
[NwsD---]    INT.MUL SL2.s32, SH84.s32, R21.s32
[NswD---]    BIT.AND.LSL R22, C80, C33, AR0
[NwsD---]    BIT.ROT.LSR R23, C33, AR0
[NswD---]      INT.ADD R26.s32, SL2.s32, R23.s32
[NwsD---]      INT.MADD R26.s32, SH34.s32, R22.s32, R26.s32
[NswD---]      INT.MADD R26.s32, SH33.s32, R26.s32, R49.s32
[NwsD---]      INT.MADD R26.s32, SH32.s32, R26.s32, R48.s32
[NswD---]      BIT.ROT.LSL R26, C33, R26
[NwsD---]      INT.ADD R27.s32, SH82.s32, R26.s32
[NwwDs--]      DMA.LD.B512 R[80:95], SH[8:9], R26, _, _, _, _,  SH[8:10]
[N--Ds--]      DMA.LD.B512 R[96:111], SH[8:9], R27, _, _, _, _,  SH[8:10]
[NswD---]    BIT.AND.LSL R24, C45, C59, AR0
[NwsD---]    BIT.ROT.LSR R25, C50, AR0
[NswD---]      BIT.AND.LSR R20, C61, SH81, R25
[NwsD---]      BIT.AND.LSL R25, C45, C111, R25
[NswD---]      INT.ADD R25.s32, R20.s32, R25.s32
[NwsD---]      INT.MADD R1.s32, SH37.s32, R24.s32, R25.s32
[NswD---]      INT.MADD R1.s32, SH36.s32, R1.s32, R42.s32
[NwsD---]      INT.MADD R1.s32, SH35.s32, R1.s32, R48.s32
[NswD---]      BIT.ROT.LSL R1, C33, R1
[NwwDs--]      DMA.LD.B64 R[234:235], SH[12:13], R1, _, _, _, _,  SH[12:14]
[NwsD---]    BIT.AND.LSL R24, C50, C111, AR0
[NswD---]    BIT.ROT.LSR R25, C61, AR0
[NwsD---]      INT.MADD R2.s32, SH40.s32, R24.s32, R25.s32
[NswD---]      INT.MADD R2.s32, SH39.s32, R2.s32, R42.s32
[NwsD---]      INT.MADD R2.s32, SH38.s32, R2.s32, R48.s32
[NswD---]      BIT.ROT.LSL R2, C33, R2
[NwsD---]    BIT.AND.LSL R40, SH71, C33, AR0
[NswD---]    BIT.ROT.LSR R41, C80, AR0
[NwsD---]    BIT.ROT.LSL R41, C46, R41
[NswD---]    BIT.AND.LSR R42, C33, C78, AR0
[NwsD---]    BIT.ROT.LSL R42, C50, R42
[NswD---]    INT.ADD R29.s32, R40.s32, R41.s32
[NwsD---]    INT.ADD R29.s32, R29.s32, R42.s32
[NswD---]      BIT.ROT.LSR R29, C33, R29
[NwsD---]      BIT.AND.LSL R28, C33, SH81, AR0
[NswD---]      BIT.ROT.LSL R20, SH68, SL3
[NwsD---]      INT.ADD R28.s32, R28.s32, R20.s32
[NswD---]      BIT.ROT.LSR R28, C33, R28
[NwsD---]    BIT.AND.LSR R40, C45, C59, AR0
[NswD---]    BIT.ROT.LSL R40, SH68, R40
[NwsD---]    BIT.ROT.LSL R41, C46, SL3
[NswD---]    BIT.AND.LSR R42, C33, C50, AR0
[NwsD---]    BIT.ROT.LSL R42, C59, R42
[NswD---]    BIT.ROT.LSR R43, C50, R0
[NwsD---]    BIT.ROT.LSL R43, C50, R43
[NswD---]    BIT.LUT.AND R20, C33, R0
[NwsD---]    BIT.ROT.LSL R20, C45, R20
[NswD---]    INT.ADD R43.s32, R43.s32, R20.s32
[NwsD---]    INT.ADD R4.s32, R40.s32, R41.s32
[NswD---]    INT.ADD R4.s32, R4.s32, R42.s32
[NwsD---]    INT.ADD R4.s32, R4.s32, R43.s32
[NswD---]      BIT.ROT.LSR R4, C33, R4
[NwsD---]    BIT.ROT.LSR R40, C46, AR0
[NswD---]    BIT.ROT.LSL R40, SH69, R40
[NwsD---]    BIT.AND.LSR R41, C33, C111, AR0
[NswD---]    BIT.ROT.LSL R41, C46, R41
[NwsD---]    BIT.LUT.AND R42, C33, R0
[NswD---]    BIT.ROT.LSL R42, C50, R42
[NwsD---]    BIT.LUT.AND R20, C34, AR0
[NswD---]    BIT.ROT.LSR R20, C61, R20
[NwsD---]    BIT.ROT.LSL R20, C61, R20
[NswD---]    INT.ADD R42.s32, R20.s32, R42.s32
[NwsD---]    INT.ADD R5.s32, R40.s32, R41.s32
[NswD---]    INT.ADD R5.s32, R5.s32, R42.s32
[NwsD---]      BIT.ROT.LSR R5, C33, R5
[NswD---]    BIT.AND.LSR R40, C80, SH76, R0
[NwsD---]    BIT.ROT.LSL R40, SH71, R40
[NswD---]    BIT.AND.LSR R41, C45, C59, R0
[NwsD---]    BIT.ROT.LSL R41, SH70, R41
[NswD---]    BIT.AND.LSR R42, C33, C50, R0
[NwsD---]    BIT.ROT.LSL R42, C59, R42
[NswD---]    BIT.AND.LSL R43, C45, C33, R0
[NwsD---]    BIT.ROT.LSR R45, C79, R0
[NswD---]    BIT.AND.LSR R50, C50, C78, R0
[NwsD---]    BIT.ROT.LSL R45, C45, R45
[NswD---]    INT.MADD R50.s32, SH77.s32, R50.s32, SL3.s32
[NwsD---]    INT.ADD R50.s32, R45.s32, R50.s32
[NswD---]    BIT.AND.LSL R44, C50, C111, R50
[NwsD---]    BIT.ROT.LSR R20, C61, R50
[NswD---]    BIT.ROT.LSL R20, C46, R20
[NwsD---]    INT.ADD R44.s32, R20.s32, R44.s32
[NswD---]    INT.ADD R6.s32, R40.s32, R44.s32
[NwsD---]    INT.ADD R6.s32, R6.s32, R41.s32
[NswD---]    INT.ADD R6.s32, R6.s32, R42.s32
[NwsD---]    INT.ADD R6.s32, R6.s32, R43.s32
[NswD---]      BIT.ROT.LSR R6, C33, R6
[NwsD---]    INT.ADD R7.s32, C0.s32, R28.s32
[NswD---]    BIT.AND.LSR R40, C80, SH76, R0
[NwsD---]    BIT.ROT.LSL R40, C50, R40
[NswD---]    BIT.LUT.AND R20, C59, R0
[NwsD---]    INT.ADD R40.s32, R40.s32, R20.s32
[NswD---]    INT.MADD R8.s32, C39.s32, R40.s32, R50.s32
[NwsD---]    BIT.ROT.LSR R9, C33, R0
[NswD---]    BIT.AND.LSL R20, C45, C33, R0
[NwsD---]    INT.MADD R9.s32, SH77.s32, R9.s32, SL3.s32
[NswD---]    INT.MADD R9.s32, C39.s32, R20.s32, R9.s32
[NwsD---]    INT.MADD R10.s32, C45.s32, C0.s32, R50.s32
[NswD---]    INT.MADD R11.s32, C45.s32, C39.s32, R50.s32
[NwsD---]    INT.MADD R12.s32, C45.s32, SH72.s32, R50.s32
[NswD---]    INT.MADD R13.s32, C45.s32, SH74.s32, R50.s32
[NwsD---]    INT.MADD R14.s32, C45.s32, SH77.s32, R50.s32
[NswD---]    INT.MADD R15.s32, C45.s32, SH78.s32, R50.s32
[NwsD---]    INT.MADD R16.s32, C45.s32, SH79.s32, R50.s32
[NswD---]    INT.MADD R17.s32, C45.s32, SH80.s32, R50.s32
[NwsD---]      BIT.ROT.LSL R3, C33, R0
[NswD---]      BIT.ROT.LSR R3, C33, R3
[NwsD---]      INT.ADD SL0.s32, SH29.s32, SH76.s32
[NswD---]      BIT.ROT.LSR SL0, C79, SL0
[NwsD---]      INT.ADD SL1.s32, C0.s32, SL0.s32
[NswD---]      BIT.LUT.AND R19, SH76, SH29
[NwsD---]      INT.BYP.TE p0, R19.s32, C0.s32
[NswD---]      INT.BYP.MOVC.PRB R19.s32, SH77.s32, R19.s32, p0
[NwsD---]      BIT.AND.LSL R18, C45, C59, R0
[NswD---]      BIT.AND.LSR R20, C80, SH76, R0
[NwsD---]      BIT.ROT.LSL R20, C80, R20
[NswD---]      INT.ADD R18.s32, R20.s32, R18.s32
[NwsD---]      INT.ADD R20.s32, C0.s32, R18.s32,
[NswD---]      INT.BYP.TL p0, R20.s32, R19.s32
[NwsD---]      INT.BYP.MOVC.PRB R238.s32, C0.s32, SH85.s32, p0
[NswD---]      INT.ADD R20.s32, C33.s32, R18.s32,
[NwsD---]      INT.BYP.TL p0, R20.s32, R19.s32
[NswD---]      INT.BYP.MOVC.PRB R239.s32, C0.s32, SH85.s32, p0
[NwsD---]      INT.ADD R20.s32, C45.s32, R18.s32,
[NswD---]      INT.BYP.TL p0, R20.s32, R19.s32
[NwsD---]      INT.BYP.MOVC.PRB R240.s32, C0.s32, SH85.s32, p0
[NswD---]      INT.ADD R20.s32, C50.s32, R18.s32,
[NwsD---]      INT.BYP.TL p0, R20.s32, R19.s32
[NswD---]      INT.BYP.MOVC.PRB R241.s32, C0.s32, SH85.s32, p0
[NwsD---]      BIT.AND.LSL R38, C33, C50, R0
[NswD---]      BIT.ROT.LSR R20, C79, R0
[NwsD---]      BIT.ROT.LSL R20, C50, R20
[NswD---]      INT.ADD R38.s32, R38.s32, R20.s32
[NwsD---]      INT.ADD R39.s32, C0.s32, R38.s32
[NswD---]      INT.BYP.TL p0, R39.s32, SH30.s32
[NwsD---]      INT.BYP.MOVC.PRB R40.s32, C52.s32, C0.s32, p0
[NswD---]      INT.BYP.TL p0, R39.s32, SH31.s32
[NwsD---]      INT.BYP.MOVC.PRB R40.s32, C36.s32, R40.s32, p0
[NswD---]      INT.ADD R39.s32, C39.s32, R38.s32
[NwsD---]      INT.BYP.TL p0, R39.s32, SH30.s32
[NswD---]      INT.BYP.MOVC.PRB R41.s32, C52.s32, C0.s32, p0
[NwsD---]      INT.BYP.TL p0, R39.s32, SH31.s32
[NswD---]      INT.BYP.MOVC.PRB R41.s32, C36.s32, R41.s32, p0
[NwsD---]      INT.ADD R39.s32, SH72.s32, R38.s32
[NswD---]      INT.BYP.TL p0, R39.s32, SH30.s32
[NwsD---]      INT.BYP.MOVC.PRB R42.s32, C52.s32, C0.s32, p0
[NswD---]      INT.BYP.TL p0, R39.s32, SH31.s32
[NwsD---]      INT.BYP.MOVC.PRB R42.s32, C36.s32, R42.s32, p0
[NswD---]      INT.ADD R39.s32, SH74.s32, R38.s32
[NwsD---]      INT.BYP.TL p0, R39.s32, SH30.s32
[NswD---]      INT.BYP.MOVC.PRB R43.s32, C52.s32, C0.s32, p0
[NwsD---]      INT.BYP.TL p0, R39.s32, SH31.s32
[NswD---]      INT.BYP.MOVC.PRB R43.s32, C36.s32, R43.s32, p0
[NwsD---]      INT.ADD R39.s32, SH77.s32, R38.s32
[NswD---]      INT.BYP.TL p0, R39.s32, SH30.s32
[NwsD---]      INT.BYP.MOVC.PRB R44.s32, C52.s32, C0.s32, p0
[NswD---]      INT.BYP.TL p0, R39.s32, SH31.s32
[NwsD---]      INT.BYP.MOVC.PRB R44.s32, C36.s32, R44.s32, p0
[NswD---]      INT.ADD R39.s32, SH78.s32, R38.s32
[NwsD---]      INT.BYP.TL p0, R39.s32, SH30.s32
[NswD---]      INT.BYP.MOVC.PRB R45.s32, C52.s32, C0.s32, p0
[NwsD---]      INT.BYP.TL p0, R39.s32, SH31.s32
[NswD---]      INT.BYP.MOVC.PRB R45.s32, C36.s32, R45.s32, p0
[NwsD---]      INT.ADD R39.s32, SH79.s32, R38.s32
[NswD---]      INT.BYP.TL p0, R39.s32, SH30.s32
[NwsD---]      INT.BYP.MOVC.PRB R46.s32, C52.s32, C0.s32, p0
[NswD---]      INT.BYP.TL p0, R39.s32, SH31.s32
[NwsD---]      INT.BYP.MOVC.PRB R46.s32, C36.s32, R46.s32, p0
[NswD---]      INT.ADD R39.s32, SH80.s32, R38.s32
[NwsD---]      INT.BYP.TL p0, R39.s32, SH30.s32
[NswD---]      INT.BYP.MOVC.PRB R47.s32, C52.s32, C0.s32, p0
[NwsD---]      INT.BYP.TL p0, R39.s32, SH31.s32
[NswD---]      INT.BYP.MOVC.PRB R47.s32, C36.s32, R47.s32, p0
[NwsD---]      FOP.MOV R112, C0
[NswD---]      FOP.MOV R113, C0
[NwsD---]      FOP.MOV R114, C0
[NswD---]      FOP.MOV R115, C0
[NwsD---]      FOP.MOV R116, C0
[NswD---]      FOP.MOV R117, C0
[NwsD---]      FOP.MOV R118, C0
[NswD---]      FOP.MOV R119, C0
[NwsD---]      FOP.MOV R120, C0
[NswD---]      FOP.MOV R121, C0
[NwsD---]      FOP.MOV R122, C0
[NswD---]      FOP.MOV R123, C0
[NwsD---]      FOP.MOV R124, C0
[NswD---]      FOP.MOV R125, C0
[NwsD---]      FOP.MOV R126, C0
[NswD---]      FOP.MOV R127, C0
[NwsD---]      FOP.MOV R128, C0
[NswD---]      FOP.MOV R129, C0
[NwsD---]      FOP.MOV R130, C0
[NswD---]      FOP.MOV R131, C0
[NwsD---]      FOP.MOV R132, C0
[NswD---]      FOP.MOV R133, C0
[NwsD---]      FOP.MOV R134, C0
[NswD---]      FOP.MOV R135, C0
[NwsD---]      FOP.MOV R136, C0
[NswD---]      FOP.MOV R137, C0
[NwsD---]      FOP.MOV R138, C0
[NswD---]      FOP.MOV R139, C0
[NwsD---]      FOP.MOV R140, C0
[NswD---]      FOP.MOV R141, C0
[NwsD---]      FOP.MOV R142, C0
[NswD---]      FOP.MOV R143, C0
[NwsD---]      FOP.MOV R144, C0
[NswD---]      FOP.MOV R145, C0
[NwsD---]      FOP.MOV R146, C0
[NswD---]      FOP.MOV R147, C0
[NwsD---]      FOP.MOV R148, C0
[NswD---]      FOP.MOV R149, C0
[NwsD---]      FOP.MOV R150, C0
[NswD---]      FOP.MOV R151, C0
[NwsD---]      FOP.MOV R152, C0
[NswD---]      FOP.MOV R153, C0
[NwsD---]      FOP.MOV R154, C0
[NswD---]      FOP.MOV R155, C0
[NwsD---]      FOP.MOV R156, C0
[NswD---]      FOP.MOV R157, C0
[NwsD---]      FOP.MOV R158, C0
[NswD---]      FOP.MOV R159, C0
[NwsD---]      FOP.MOV R160, C0
[NswD---]      FOP.MOV R161, C0
[NwsD---]      FOP.MOV R162, C0
[NswD---]      FOP.MOV R163, C0
[NwsD---]      FOP.MOV R164, C0
[NswD---]      FOP.MOV R165, C0
[NwsD---]      FOP.MOV R166, C0
[NswD---]      FOP.MOV R167, C0
[NwsD---]      FOP.MOV R168, C0
[NswD---]      FOP.MOV R169, C0
[NwsD---]      FOP.MOV R170, C0
[NswD---]      FOP.MOV R171, C0
[NwsD---]      FOP.MOV R172, C0
[NswD---]      FOP.MOV R173, C0
[NwsD---]      FOP.MOV R174, C0
[NswD---]      FOP.MOV R175, C0
[NwsD---]      FOP.MOV R225, C0
[NwwDw--]      CTRL.NOP
[NwwD-s-]       LDMA.WR.B128.BST1 R[80:83], R29, X4, 128 , BURST_STRIDE=16
[N--D-s-]       LDMA.WR.B128.BST1 R[84:87], R29, X4, 384 , BURST_STRIDE=16
[N--D-s-]       LDMA.WR.B128.BST1 R[88:91], R29, X4, 8320 , BURST_STRIDE=16
[N--D-o-]       LDMA.WR.B128.BST1 R[92:95], R29, X4, 8576 , BURST_STRIDE=16
[NwwD---]       AP.BARRIER LM0
[N--D-s-]       LDMA.RD.B32.BST4  R[176:179],    R28, X4, 128, BURST_STRIDE=512
[N--D-s-]       LDMA.RD.B32.BST4  R[180:183],    R28, X4, 8320, BURST_STRIDE=512
[N--D-s-]       LDMA.RD.B32.BST4  R[184:187],   R28, X4, 16512, BURST_STRIDE=512
[N--D-s-]       LDMA.RD.B32.BST4  R[188:191],  R28, X4, 24704, BURST_STRIDE=512
[N--D-s-]       LDMA.WR.B128.BST1 R[96:99], R29, X4, 128 , BURST_STRIDE=16
[N--D-s-]       LDMA.WR.B128.BST1 R[100:103], R29, X4, 384 , BURST_STRIDE=16
[N--D-s-]       LDMA.WR.B128.BST1 R[104:107], R29, X4, 8320 , BURST_STRIDE=16
[N--D-o-]       LDMA.WR.B128.BST1 R[108:111], R29, X4, 8576 , BURST_STRIDE=16
[NwwD---]       AP.BARRIER LM0
[N--D-s-]       LDMA.RD.B32.BST4  R[192:195],  R28, X4, 128, BURST_STRIDE=512
[N--D-s-]       LDMA.RD.B32.BST4  R[196:199],  R28, X4, 8320, BURST_STRIDE=512
[N--D-s-]       LDMA.RD.B32.BST4  R[200:203],  R28, X4, 16512, BURST_STRIDE=512
[N--D-s-]       LDMA.RD.B32.BST4  R[204:207],  R28, X4, 24704, BURST_STRIDE=512
[NwwD-s-]       LDMA.WR.B64.BST1 R[234:235], R4, X4, 32896 , BURST_STRIDE=16
[NswD-w-]       INT.ADD R1.s32, SH77.s32, R1.s32
[NwwD---]       AP.BARRIER LM0
[N--D-o-]       LDMA.RD.B32.BST4  R[209:212], R3, X4, 32896, BURST_STRIDE=512
[N-wD---]       BIT.LUT.AND R176, R40, R176
[N--D---]       BIT.LUT.AND R177, R40, R177
[N--D---]       BIT.LUT.AND R178, R40, R178
[N--D---]       BIT.LUT.AND R179, R40, R179
[N--D---]       BIT.LUT.AND R180, R41, R180
[N--D---]       BIT.LUT.AND R181, R41, R181
[N--D---]       BIT.LUT.AND R182, R41, R182
[N--D---]       BIT.LUT.AND R183, R41, R183
[N--D---]       BIT.LUT.AND R184, R42, R184
[N--D---]       BIT.LUT.AND R185, R42, R185
[N--D---]       BIT.LUT.AND R186, R42, R186
[N--D---]       BIT.LUT.AND R187, R42, R187
[N--D---]       BIT.LUT.AND R188, R43, R188
[N--D---]       BIT.LUT.AND R189, R43, R189
[N--D---]       BIT.LUT.AND R190, R43, R190
[N--D---]       BIT.LUT.AND R191, R43, R191
[N--D---]       BIT.LUT.AND R192, R44, R192
[N--D---]       BIT.LUT.AND R193, R44, R193
[N--D---]       BIT.LUT.AND R194, R44, R194
[N--D---]       BIT.LUT.AND R195, R44, R195
[N--D---]       BIT.LUT.AND R196, R45, R196
[N--D---]       BIT.LUT.AND R197, R45, R197
[N--D---]       BIT.LUT.AND R198, R45, R198
[N--D---]       BIT.LUT.AND R199, R45, R199
[N--D---]       BIT.LUT.AND R200, R46, R200
[N--D---]       BIT.LUT.AND R201, R46, R201
[N--D---]       BIT.LUT.AND R202, R46, R202
[N--D---]       BIT.LUT.AND R203, R46, R203
[N--D---]       BIT.LUT.AND R204, R47, R204
[N--D---]       BIT.LUT.AND R205, R47, R205
[N--D---]       BIT.LUT.AND R206, R47, R206
[N-sD---]       BIT.LUT.AND R207, R47, R207
[NswD---]      FOP.MOV I0, SH85
; First MMA1 (1) QK = GEMM_TN(Q, K)
[NwwDs--]      DMA.LD.B64 R[234:235], SH[12:13], R1, _, _, _, _,  SH[12:14]
[Nw-D---] MMA.323216 R48[4:2][2:1].f32, R176[2:1][1:1].bf16.row, R209[1:1][2:1].bf16.col, 0
[N--D---] MMA.323216 R56[4:2][2:1].f32, R176[2:1][1:1].bf16.row, R211[1:1][2:1].bf16.col, 0
[Nw-D-s-]       LDMA.RD.B32.BST4  R[217:220], R3, X4, 34944, BURST_STRIDE=512
[N--D---] MMA.323216 R64[4:2][2:1].f32, R178[2:1][1:1].bf16.row, R209[1:1][2:1].bf16.col, 0
[Ns-D-w-] MMA.323216 R72[4:2][2:1].f32, R178[2:1][1:1].bf16.row, R211[1:1][2:1].bf16.col, 0
[N--D---] MMA.323216 R48[4:2][2:1].f32, R180[2:1][1:1].bf16.row, R217[1:1][2:1].bf16.col, R48[4:2][2:1].f32
[NwwD---]       AP.BARRIER LM0
[N--Dw--]       CTRL.NOP
[N--D-s-]       LDMA.WR.B64.BST1 R[234:235], R4, X4, 32896 , BURST_STRIDE=16
[Ns-D-w-]       INT.ADD R1.s32, SH77.s32, R1.s32
[Nw-D---]       AP.BARRIER LM0
[N--D---] MMA.323216 R56[4:2][2:1].f32, R180[2:1][1:1].bf16.row, R219[1:1][2:1].bf16.col, R56[4:2][2:1].f32
[Nw-D-s-]       LDMA.RD.B32.BST4  R[209:212], R3, X4, 32896, BURST_STRIDE=512
[N--D---] MMA.323216 R64[4:2][2:1].f32, R182[2:1][1:1].bf16.row, R217[1:1][2:1].bf16.col, R64[4:2][2:1].f32
[Ns-D-w-] MMA.323216 R72[4:2][2:1].f32, R182[2:1][1:1].bf16.row, R219[1:1][2:1].bf16.col, R72[4:2][2:1].f32
[NwwDs--]      DMA.LD.B64 R[234:235], SH[12:13], R1, _, _, _, _,  SH[12:14]
[N--D---] MMA.323216 R48[4:2][2:1].f32, R184[2:1][1:1].bf16.row, R209[1:1][2:1].bf16.col, R48[4:2][2:1].f32
[N--D---] MMA.323216 R56[4:2][2:1].f32, R184[2:1][1:1].bf16.row, R211[1:1][2:1].bf16.col, R56[4:2][2:1].f32
[Nw-D-s-]       LDMA.RD.B32.BST4  R[217:220], R3, X4, 34944, BURST_STRIDE=512
[N--D---] MMA.323216 R64[4:2][2:1].f32, R186[2:1][1:1].bf16.row, R209[1:1][2:1].bf16.col, R64[4:2][2:1].f32
[Ns-D-w-] MMA.323216 R72[4:2][2:1].f32, R186[2:1][1:1].bf16.row, R211[1:1][2:1].bf16.col, R72[4:2][2:1].f32
[N--D---] MMA.323216 R48[4:2][2:1].f32, R188[2:1][1:1].bf16.row, R217[1:1][2:1].bf16.col, R48[4:2][2:1].f32
[NwwD---]       AP.BARRIER LM0
[N--Dw--]       CTRL.NOP
[N--D-s-]       LDMA.WR.B64.BST1 R[234:235], R4, X4, 32896 , BURST_STRIDE=16
[Ns-D-w-]       INT.ADD R1.s32, SH77.s32, R1.s32
[Nw-D---]       AP.BARRIER LM0
[N--D---] MMA.323216 R56[4:2][2:1].f32, R188[2:1][1:1].bf16.row, R219[1:1][2:1].bf16.col, R56[4:2][2:1].f32
[Nw-D-s-]       LDMA.RD.B32.BST4  R[209:212], R3, X4, 32896, BURST_STRIDE=512
[N--D---] MMA.323216 R64[4:2][2:1].f32, R190[2:1][1:1].bf16.row, R217[1:1][2:1].bf16.col, R64[4:2][2:1].f32
[Ns-D-w-] MMA.323216 R72[4:2][2:1].f32, R190[2:1][1:1].bf16.row, R219[1:1][2:1].bf16.col, R72[4:2][2:1].f32
[NwwDs--]      DMA.LD.B64 R[234:235], SH[12:13], R1, _, _, _, _,  SH[12:14]
[N--D---] MMA.323216 R48[4:2][2:1].f32, R192[2:1][1:1].bf16.row, R209[1:1][2:1].bf16.col, R48[4:2][2:1].f32
[N--D---] MMA.323216 R56[4:2][2:1].f32, R192[2:1][1:1].bf16.row, R211[1:1][2:1].bf16.col, R56[4:2][2:1].f32
[Nw-D-s-]       LDMA.RD.B32.BST4  R[217:220], R3, X4, 34944, BURST_STRIDE=512
[N--D---] MMA.323216 R64[4:2][2:1].f32, R194[2:1][1:1].bf16.row, R209[1:1][2:1].bf16.col, R64[4:2][2:1].f32
[Ns-D-w-] MMA.323216 R72[4:2][2:1].f32, R194[2:1][1:1].bf16.row, R211[1:1][2:1].bf16.col, R72[4:2][2:1].f32
[N--D---] MMA.323216 R48[4:2][2:1].f32, R196[2:1][1:1].bf16.row, R217[1:1][2:1].bf16.col, R48[4:2][2:1].f32
[NwwD---]       AP.BARRIER LM0
[N--Dw--]       CTRL.NOP
[N--D-s-]       LDMA.WR.B64.BST1 R[234:235], R4, X4, 32896 , BURST_STRIDE=16
[Ns-D-w-]       INT.ADD R1.s32, SH56.s32, R1.s32
[Nw-D---]       AP.BARRIER LM0
[N--D---] MMA.323216 R56[4:2][2:1].f32, R196[2:1][1:1].bf16.row, R219[1:1][2:1].bf16.col, R56[4:2][2:1].f32
[Nw-D-s-]       LDMA.RD.B32.BST4  R[209:212], R3, X4, 32896, BURST_STRIDE=512
[N--D---] MMA.323216 R64[4:2][2:1].f32, R198[2:1][1:1].bf16.row, R217[1:1][2:1].bf16.col, R64[4:2][2:1].f32
[Ns-D-w-] MMA.323216 R72[4:2][2:1].f32, R198[2:1][1:1].bf16.row, R219[1:1][2:1].bf16.col, R72[4:2][2:1].f32
[NwwDs--]      DMA.LD.B64 R[234:235], SH[12:13], R1, _, _, _, _,  SH[12:14]
[N--D---] MMA.323216 R48[4:2][2:1].f32, R200[2:1][1:1].bf16.row, R209[1:1][2:1].bf16.col, R48[4:2][2:1].f32
[N--D---] MMA.323216 R56[4:2][2:1].f32, R200[2:1][1:1].bf16.row, R211[1:1][2:1].bf16.col, R56[4:2][2:1].f32
[Nw-D-s-]       LDMA.RD.B32.BST4  R[217:220], R3, X4, 34944, BURST_STRIDE=512
[N--D---] MMA.323216 R64[4:2][2:1].f32, R202[2:1][1:1].bf16.row, R209[1:1][2:1].bf16.col, R64[4:2][2:1].f32
[Ns-D-w-] MMA.323216 R72[4:2][2:1].f32, R202[2:1][1:1].bf16.row, R211[1:1][2:1].bf16.col, R72[4:2][2:1].f32
[N--D---] MMA.323216 R48[4:2][2:1].f32, R204[2:1][1:1].bf16.row, R217[1:1][2:1].bf16.col, R48[4:2][2:1].f32
[NwwD---]       AP.BARRIER LM0
[N--Dw--]       CTRL.NOP
[N--D-s-]       LDMA.WR.B64.BST1 R[234:235], R4, X4, 32896 , BURST_STRIDE=16
[Ns-D-w-]       INT.ADD R1.s32, SH77.s32, R1.s32
[Nw-D---]       AP.BARRIER LM0
[N--D---] MMA.323216 R56[4:2][2:1].f32, R204[2:1][1:1].bf16.row, R219[1:1][2:1].bf16.col, R56[4:2][2:1].f32
[Nw-D-s-]       LDMA.RD.B32.BST4  R[209:212], R3, X4, 32896, BURST_STRIDE=512
[N--D---] MMA.323216 R64[4:2][2:1].f32, R206[2:1][1:1].bf16.row, R217[1:1][2:1].bf16.col, R64[4:2][2:1].f32
[Ns-D-w-] MMA.323216 R72[4:2][2:1].f32, R206[2:1][1:1].bf16.row, R219[1:1][2:1].bf16.col, R72[4:2][2:1].f32
[N-sD---]       INT.BYP.TE p0, SL0.s32, C33.s32
[NwwD---]  P0  FOP.FMA R80, SH50, R48, R238
[N--D---]  P0  FOP.FMA R81, SH50, R49, R239
[NwwD---]  !P0  FOP.FMA R80, SH50, R48, C0
[N--D---]  !P0  FOP.FMA R81, SH50, R49, C0
[N--D---]  P0  FOP.FMA R82, SH50, R50, R238
[N--D---]  P0  FOP.FMA R83, SH50, R51, R239
[N--D---]  !P0  FOP.FMA R82, SH50, R50, C0
[N--D---]  !P0  FOP.FMA R83, SH50, R51, C0
[N--D---]  P0  FOP.FMA R84, SH50, R52, R238
[N--D---]  P0  FOP.FMA R85, SH50, R53, R239
[N--D---]  !P0  FOP.FMA R84, SH50, R52, C0
[N--D---]  !P0  FOP.FMA R85, SH50, R53, C0
[N--D---]  P0  FOP.FMA R86, SH50, R54, R238
[N--D---]  P0  FOP.FMA R87, SH50, R55, R239
[N--D---]  !P0  FOP.FMA R86, SH50, R54, C0
[N--D---]  !P0  FOP.FMA R87, SH50, R55, C0
[N--D---]  P0  FOP.FMA R88, SH50, R56, R240
[N--D---]  P0  FOP.FMA R89, SH50, R57, R241
[N--D---]  !P0  FOP.FMA R88, SH50, R56, C0
[N--D---]  !P0  FOP.FMA R89, SH50, R57, C0
[N--D---]  P0  FOP.FMA R90, SH50, R58, R240
[N--D---]  P0  FOP.FMA R91, SH50, R59, R241
[N--D---]  !P0  FOP.FMA R90, SH50, R58, C0
[N--D---]  !P0  FOP.FMA R91, SH50, R59, C0
[N--D---]  P0  FOP.FMA R92, SH50, R60, R240
[N--D---]  P0  FOP.FMA R93, SH50, R61, R241
[N--D---]  !P0  FOP.FMA R92, SH50, R60, C0
[N--D---]  !P0  FOP.FMA R93, SH50, R61, C0
[N--D---]  P0  FOP.FMA R94, SH50, R62, R240
[N--D---]  P0  FOP.FMA R95, SH50, R63, R241
[N--D---]  !P0  FOP.FMA R94, SH50, R62, C0
[N--D---]  !P0  FOP.FMA R95, SH50, R63, C0
[N--D---]  P0  FOP.FMA R96, SH50, R64, R238
[N--D---]  P0  FOP.FMA R97, SH50, R65, R239
[N--D---]  !P0  FOP.FMA R96, SH50, R64, C0
[N--D---]  !P0  FOP.FMA R97, SH50, R65, C0
[N--D---]  P0  FOP.FMA R98, SH50, R66, R238
[N--D---]  P0  FOP.FMA R99, SH50, R67, R239
[N--D---]  !P0  FOP.FMA R98, SH50, R66, C0
[N--D---]  !P0  FOP.FMA R99, SH50, R67, C0
[N--D---]  P0  FOP.FMA R100, SH50, R68, R238
[N--D---]  P0  FOP.FMA R101, SH50, R69, R239
[N--D---]  !P0  FOP.FMA R100, SH50, R68, C0
[N--D---]  !P0  FOP.FMA R101, SH50, R69, C0
[N--D---]  P0  FOP.FMA R102, SH50, R70, R238
[N--D---]  P0  FOP.FMA R103, SH50, R71, R239
[N--D---]  !P0  FOP.FMA R102, SH50, R70, C0
[N--D---]  !P0  FOP.FMA R103, SH50, R71, C0
[N--D---]  P0  FOP.FMA R104, SH50, R72, R240
[N--D---]  P0  FOP.FMA R105, SH50, R73, R241
[N--D---]  !P0  FOP.FMA R104, SH50, R72, C0
[N--D---]  !P0  FOP.FMA R105, SH50, R73, C0
[N--D---]  P0  FOP.FMA R106, SH50, R74, R240
[N--D---]  P0  FOP.FMA R107, SH50, R75, R241
[N--D---]  !P0  FOP.FMA R106, SH50, R74, C0
[N--D---]  !P0  FOP.FMA R107, SH50, R75, C0
[N--D---]  P0  FOP.FMA R108, SH50, R76, R240
[N--D---]  P0  FOP.FMA R109, SH50, R77, R241
[N--D---]  !P0  FOP.FMA R108, SH50, R76, C0
[N--D---]  !P0  FOP.FMA R109, SH50, R77, C0
[N--D---]  P0  FOP.FMA R110, SH50, R78, R240
[N-sD--w]  P0  FOP.FMA R111, SH50, R79, R241
[N--D---]  !P0  FOP.FMA R110, SH50, R78, C0
[N-sD---]  !P0  FOP.FMA R111, SH50, R79, C0
; SOFTMAX MAX REDUCE BEG
[NwwD---]      FOP.CMP_MOVC.MAX.TST R30, R80, R81
[N--D---]      FOP.CMP_MOVC.MAX.TST R31, R82, R83
[N--D---]      FOP.CMP_MOVC.MAX.TST R32, R84, R85
[N--D---]      FOP.CMP_MOVC.MAX.TST R33, R86, R87
[N--D---]      FOP.CMP_MOVC.MAX.TST R34, R96, R97
[N--D---]      FOP.CMP_MOVC.MAX.TST R35, R98, R99
[N--D---]      FOP.CMP_MOVC.MAX.TST R36, R100, R101
[N--D---]      FOP.CMP_MOVC.MAX.TST R37, R102, R103
[N--D---]      FOP.CMP_MOVC.MAX.TST R30, R30, R88
[N--D---]      FOP.CMP_MOVC.MAX.TST R31, R31, R90
[N--D---]      FOP.CMP_MOVC.MAX.TST R32, R32, R92
[N--D---]      FOP.CMP_MOVC.MAX.TST R33, R33, R94
[N--D---]      FOP.CMP_MOVC.MAX.TST R34, R34, R104
[N--D---]      FOP.CMP_MOVC.MAX.TST R35, R35, R106
[N--D---]      FOP.CMP_MOVC.MAX.TST R36, R36, R108
[N--D---]      FOP.CMP_MOVC.MAX.TST R37, R37, R110
[N--D---]      FOP.CMP_MOVC.MAX.TST R30, R30, R89
[N--D---]      FOP.CMP_MOVC.MAX.TST R31, R31, R91
[N--D---]      FOP.CMP_MOVC.MAX.TST R32, R32, R93
[N--D---]      FOP.CMP_MOVC.MAX.TST R33, R33, R95
[N--D---]      FOP.CMP_MOVC.MAX.TST R34, R34, R105
[N--D---]      FOP.CMP_MOVC.MAX.TST R35, R35, R107
[N--D---]      FOP.CMP_MOVC.MAX.TST R36, R36, R109
[N-sD---]      FOP.CMP_MOVC.MAX.TST R37, R37, R111
[N-wD-s-]      LDMA.WR.B32.BST4 R[30:33], R8, X4, 41088 , BURST_STRIDE=512
[N--D-s-]      LDMA.WR.B32.BST4 R[34:37], R8, X4, 43136 , BURST_STRIDE=512
[NwwD-w-]    CTRL.NOP
[Nw-D---]      LDMA.RD.B128.BST1 R[30:33], R9, X4, 41088 , BURST_STRIDE=16
[N-sD---]      LDMA.RD.B128.BST1 R[34:37], R9, X4, 41120 , BURST_STRIDE=16
[N-wD---]      FOP.CMP_MOVC.MAX.TST R30, R30, R31
[N--D---]      FOP.CMP_MOVC.MAX.TST R32, R32, R33
[N--D---]      FOP.CMP_MOVC.MAX.TST R34, R34, R35
[N--D---]      FOP.CMP_MOVC.MAX.TST R36, R36, R37
[N--D---]      FOP.CMP_MOVC.MAX.TST R30, R30, R32
[N--D---]      FOP.CMP_MOVC.MAX.TST R34, R34, R36
[N--D---]      FOP.CMP_MOVC.MAX.TST R30, R30, R34
[N--D---]      FOP.SWPX R31, R30
[N--D---]      FOP.CMP_MOVC.MAX.TST R30, R30, R31
[N--D---]      FOP.CMP_MOVC.MAX.TST R19, R30, I0
[N-sD---]      FOP.ADD R18, I0, R19.neg
[N-wD--s]      AF32.EXP2 R18, R18
[N-wD--w]      SHUFFLE.IDX R30, R19, R10, SH81
[N--D---]      SHUFFLE.IDX R242, R18, R10, SH81
[N--D---]      SHUFFLE.IDX R31, R19, R11, SH81
[N--D---]      SHUFFLE.IDX R243, R18, R11, SH81
[N--D---]      SHUFFLE.IDX R32, R19, R12, SH81
[N--D---]      SHUFFLE.IDX R208, R18, R12, SH81
[N--D---]      SHUFFLE.IDX R33, R19, R13, SH81
[N--D---]      SHUFFLE.IDX AR1, R18, R13, SH81
[N--D---]      SHUFFLE.IDX R34, R19, R14, SH81
[N--D---]      SHUFFLE.IDX AR2, R18, R14, SH81
[N--D---]      SHUFFLE.IDX R35, R19, R15, SH81
[N--D---]      SHUFFLE.IDX AR3, R18, R15, SH81
[N--D---]      SHUFFLE.IDX R36, R19, R16, SH81
[N--D---]      SHUFFLE.IDX AR4, R18, R16, SH81
[N--D---]      SHUFFLE.IDX R37, R19, R17, SH81
[N-sD---]      SHUFFLE.IDX AR5, R18, R17, SH81
[N-wD---]      FOP.ADD R80, R80, R30.neg
[N--D---]      FOP.ADD R81, R81, R30.neg
[N--D---]      FOP.ADD R88, R88, R30.neg
[N--D---]      FOP.ADD R89, R89, R30.neg
[N--D---]      FOP.ADD R82, R82, R31.neg
[N--D---]      FOP.ADD R83, R83, R31.neg
[N--D---]      FOP.ADD R90, R90, R31.neg
[N--D---]      FOP.ADD R91, R91, R31.neg
[N--D---]      FOP.ADD R84, R84, R32.neg
[N--D---]      FOP.ADD R85, R85, R32.neg
[N--D---]      FOP.ADD R92, R92, R32.neg
[N--D---]      FOP.ADD R93, R93, R32.neg
[N--D---]      FOP.ADD R86, R86, R33.neg
[N--D---]      FOP.ADD R87, R87, R33.neg
[N--D---]      FOP.ADD R94, R94, R33.neg
[N--D---]      FOP.ADD R95, R95, R33.neg
[N--D---]      FOP.ADD R96, R96, R34.neg
[N--D---]      FOP.ADD R97, R97, R34.neg
[N--D---]      FOP.ADD R104, R104, R34.neg
[N--D---]      FOP.ADD R105, R105, R34.neg
[N--D---]      FOP.ADD R98, R98, R35.neg
[N--D---]      FOP.ADD R99, R99, R35.neg
[N--D---]      FOP.ADD R106, R106, R35.neg
[N--D---]      FOP.ADD R107, R107, R35.neg
[N--D---]      FOP.ADD R100, R100, R36.neg
[N--D---]      FOP.ADD R101, R101, R36.neg
[N--D---]      FOP.ADD R108, R108, R36.neg
[N--D---]      FOP.ADD R109, R109, R36.neg
[N--D---]      FOP.ADD R102, R102, R37.neg
[N--D---]      FOP.ADD R103, R103, R37.neg
[N--D---]      FOP.ADD R110, R110, R37.neg
[N-sD---]      FOP.ADD R111, R111, R37.neg
; SOFTMAX MAX REDUCE END
[NwwD---]    CTRL.NOP
[NwsD---]      INT.ADD SL1.s32, C33.neg.s32, SL1.s32
[NswD---]      INT.ADD SL0.s32, C33.neg.s32, SL0.s32
[NswD---]       INT.BYP.TG p0, SL1.s32, C0.s32
[Nw-D---]       CTRL.BR.ALLP0F LAST_LOOP
MAIN_LOOP:
[Nw-Ds--]      DMA.LD.B64 R[234:235], SH[12:13], R1, _, _, _, _,  SH[12:14]
; SOFTMAX UPDATE ACC BEG
[N-wD---]      AF32.EXP2 R80, R80
[NwwD---]      FOP.MUL R112, R112, R242
[N--D---]      AF32.EXP2 R81, R81
[N--D---]      FOP.MUL R113, R113, R242
[N--D---]      AF32.EXP2 R88, R88
[N--D---]      FOP.MUL R120, R120, R242
[N--D---]      AF32.EXP2 R89, R89
[N--D---] MMA.323216 R48[4:2][2:1].f32, R176[2:1][1:1].bf16.row, R209[1:1][2:1].bf16.col, 0
[N--D---]      FOP.MUL R121, R121, R242
[N--D---]      FOP.MUL R128, R128, R242
[N--D---]      FOP.MUL R129, R129, R242
[N--D---]      FOP.MUL R136, R136, R242
[N--D---]      FOP.MUL R137, R137, R242
[N--D---]      AF32.EXP2 R82, R82
[N--D---] MMA.323216 R56[4:2][2:1].f32, R176[2:1][1:1].bf16.row, R211[1:1][2:1].bf16.col, 0
[Nw-D-s-]       LDMA.RD.B32.BST4  R[217:220], R3, X4, 34944, BURST_STRIDE=512
[N--D---]      FOP.MUL R114, R114, R243
[N--D---]      AF32.EXP2 R83, R83
[N--D---]      FOP.MUL R115, R115, R243
[N--D---]      AF32.EXP2 R90, R90
[N--D---]      FOP.MUL R122, R122, R243
[N--D---]      AF32.EXP2 R91, R91
[N--D---]      FOP.MUL R123, R123, R243
[N--D---]      FOP.MUL R130, R130, R243
[N--D---] MMA.323216 R64[4:2][2:1].f32, R178[2:1][1:1].bf16.row, R209[1:1][2:1].bf16.col, 0
[N--D---]      FOP.MUL R131, R131, R243
[N--D---]      FOP.MUL R138, R138, R243
[N--D---]      FOP.MUL R139, R139, R243
[N--D---]      AF32.EXP2 R84, R84
[N--D---]      FOP.MUL R116, R116, R208
[N--D---]      AF32.EXP2 R85, R85
[N--D---]      FOP.MUL R117, R117, R208
[N--D---]      AF32.EXP2 R92, R92
[Ns-D-w-] MMA.323216 R72[4:2][2:1].f32, R178[2:1][1:1].bf16.row, R211[1:1][2:1].bf16.col, 0
[N--D---]      FOP.MUL R124, R124, R208
[N--D---]      AF32.EXP2 R93, R93
[N--D---]      FOP.MUL R125, R125, R208
[N--D---]      FOP.MUL R132, R132, R208
[N--D---]      FOP.MUL R133, R133, R208
[N--D---]      FOP.MUL R140, R140, R208
[N--D---]      FOP.MUL R141, R141, R208
[N--D---]      AF32.EXP2 R86, R86
[N--D---] MMA.323216 R48[4:2][2:1].f32, R180[2:1][1:1].bf16.row, R217[1:1][2:1].bf16.col, R48[4:2][2:1].f32
[NwwD---]       AP.BARRIER LM0
[N--Dw--]       CTRL.NOP
[N--D-s-]       LDMA.WR.B64.BST1 R[234:235], R4, X4, 32896 , BURST_STRIDE=16
[Ns-D-w-]       INT.ADD R1.s32, SH77.s32, R1.s32
[Nw-D---]       AP.BARRIER LM0
[N--D---]      FOP.MUL R118, R118, AR1
[N--D---]      AF32.EXP2 R87, R87
[N--D---]      FOP.MUL R119, R119, AR1
[N--D---]      AF32.EXP2 R94, R94
[N--D---]      FOP.MUL R126, R126, AR1
[N--D---]      AF32.EXP2 R95, R95
[N--D---] MMA.323216 R56[4:2][2:1].f32, R180[2:1][1:1].bf16.row, R219[1:1][2:1].bf16.col, R56[4:2][2:1].f32
[Nw-D-s-]       LDMA.RD.B32.BST4  R[209:212], R3, X4, 32896, BURST_STRIDE=512
[N--D---]      FOP.MUL R127, R127, AR1
[N--D---]      FOP.MUL R134, R134, AR1
[N--D---]      FOP.MUL R135, R135, AR1
[N--D---]      FOP.MUL R142, R142, AR1
[N--D---]      FOP.MUL R143, R143, AR1
[N--D---]      AF32.EXP2 R96, R96
[N--D---]      FOP.MUL R144, R144, AR2
[N--D---]      AF32.EXP2 R97, R97
[N--D---] MMA.323216 R64[4:2][2:1].f32, R182[2:1][1:1].bf16.row, R217[1:1][2:1].bf16.col, R64[4:2][2:1].f32
[N--D---]      FOP.MUL R145, R145, AR2
[N--D---]      AF32.EXP2 R104, R104
[N--D---]      FOP.MUL R152, R152, AR2
[N--D---]      AF32.EXP2 R105, R105
[N--D---]      FOP.MUL R153, R153, AR2
[N--D---]      FOP.MUL R160, R160, AR2
[N--D---]      FOP.MUL R161, R161, AR2
[N--D---]      FOP.MUL R168, R168, AR2
[Ns-D-w-] MMA.323216 R72[4:2][2:1].f32, R182[2:1][1:1].bf16.row, R219[1:1][2:1].bf16.col, R72[4:2][2:1].f32
[NwwDs--]      DMA.LD.B64 R[234:235], SH[12:13], R1, _, _, _, _,  SH[12:14]
[N--D---]      FOP.MUL R169, R169, AR2
[N--D---]      AF32.EXP2 R98, R98
[N--D---]      FOP.MUL R146, R146, AR3
[N--D---]      AF32.EXP2 R99, R99
[N--D---]      FOP.MUL R147, R147, AR3
[N--D---]      AF32.EXP2 R106, R106
[N--D---]      FOP.MUL R154, R154, AR3
[N--D---]      AF32.EXP2 R107, R107
[N--D---] MMA.323216 R48[4:2][2:1].f32, R184[2:1][1:1].bf16.row, R209[1:1][2:1].bf16.col, R48[4:2][2:1].f32
[N--D---]      FOP.MUL R155, R155, AR3
[N--D---]      FOP.MUL R162, R162, AR3
[N--D---]      FOP.MUL R163, R163, AR3
[N--D---]      FOP.MUL R170, R170, AR3
[N--D---]      FOP.MUL R171, R171, AR3
[N--D---]      AF32.EXP2 R100, R100
[N--D---] MMA.323216 R56[4:2][2:1].f32, R184[2:1][1:1].bf16.row, R211[1:1][2:1].bf16.col, R56[4:2][2:1].f32
[Nw-D-s-]       LDMA.RD.B32.BST4  R[217:220], R3, X4, 34944, BURST_STRIDE=512
[N--D---]      FOP.MUL R148, R148, AR4
[N--D---]      AF32.EXP2 R101, R101
[N--D---]      FOP.MUL R149, R149, AR4
[N--D---]      AF32.EXP2 R108, R108
[N--D---]      FOP.MUL R156, R156, AR4
[N--D---]      AF32.EXP2 R109, R109
[N--D---]      FOP.MUL R157, R157, AR4
[N--D---]      FOP.MUL R164, R164, AR4
[N--D---] MMA.323216 R64[4:2][2:1].f32, R186[2:1][1:1].bf16.row, R209[1:1][2:1].bf16.col, R64[4:2][2:1].f32
[N--D---]      FOP.MUL R165, R165, AR4
[N--D---]      FOP.MUL R172, R172, AR4
[N--D---]      FOP.MUL R173, R173, AR4
[N--D---]      AF32.EXP2 R102, R102
[N--D---]      FOP.MUL R150, R150, AR5
[N--D---]      AF32.EXP2 R103, R103
[N--D---]      FOP.MUL R151, R151, AR5
[N--D---]      AF32.EXP2 R110, R110
[Ns-D-w-] MMA.323216 R72[4:2][2:1].f32, R186[2:1][1:1].bf16.row, R211[1:1][2:1].bf16.col, R72[4:2][2:1].f32
[N--D---]      FOP.MUL R158, R158, AR5
[N--D--s]      AF32.EXP2 R111, R111
[N--D---]      FOP.MUL R159, R159, AR5
[N--D---]      FOP.MUL R166, R166, AR5
[N--D---]      FOP.MUL R167, R167, AR5
[N--D---]      FOP.MUL R174, R174, AR5
[N--D--w]      FOP.MUL R175, R175, AR5
; SOFTMAX UPDATE ACC END
[N--D---] MMA.323216 R48[4:2][2:1].f32, R188[2:1][1:1].bf16.row, R217[1:1][2:1].bf16.col, R48[4:2][2:1].f32
[NwwD---]       AP.BARRIER LM0
[N--Dw--]       CTRL.NOP
[N--D-s-]       LDMA.WR.B64.BST1 R[234:235], R4, X4, 32896 , BURST_STRIDE=16
[Ns-D-w-]       INT.ADD R1.s32, SH77.s32, R1.s32
[Nw-D---]       AP.BARRIER LM0
; SOFTMAX Reg2Sh BEG
[N--D---]      FOP.ADD R32.l.bf16, C0, R80
[N--D---]      FOP.ADD R32.h.bf16, C0, R81
[N--D---]      FOP.ADD R33.l.bf16, C0, R88
[N--D---]      FOP.ADD R33.h.bf16, C0, R89
[N--D---]      FOP.ADD R34.l.bf16, C0, R82
[N--D---] MMA.323216 R56[4:2][2:1].f32, R188[2:1][1:1].bf16.row, R219[1:1][2:1].bf16.col, R56[4:2][2:1].f32
[Nw-D-s-]       LDMA.RD.B32.BST4  R[209:212], R3, X4, 32896, BURST_STRIDE=512
[N--D---]      FOP.ADD R34.h.bf16, C0, R83
[N--D---]      FOP.ADD R35.l.bf16, C0, R90
[N--D---]      FOP.ADD R35.h.bf16, C0, R91
[N--D---]      FOP.ADD R36.l.bf16, C0, R84
[N--D---]      FOP.ADD R36.h.bf16, C0, R85
[N--D---]      FOP.ADD R37.l.bf16, C0, R92
[N--D---]      FOP.ADD R37.h.bf16, C0, R93
[N--D---]      FOP.ADD R38.l.bf16, C0, R86
[N--D---] MMA.323216 R64[4:2][2:1].f32, R190[2:1][1:1].bf16.row, R217[1:1][2:1].bf16.col, R64[4:2][2:1].f32
[N--D---]      FOP.ADD R38.h.bf16, C0, R87
[N--D---]      FOP.ADD R39.l.bf16, C0, R94
[N--D---]      FOP.ADD R39.h.bf16, C0, R95
[N--D---]      FOP.ADD R40.l.bf16, C0, R96
[N--D---]      FOP.ADD R40.h.bf16, C0, R97
[N--D---]      FOP.ADD R41.l.bf16, C0, R104
[N--D---]      FOP.ADD R41.h.bf16, C0, R105
[N--D---]      FOP.ADD R42.l.bf16, C0, R98
[Ns-D-w-] MMA.323216 R72[4:2][2:1].f32, R190[2:1][1:1].bf16.row, R219[1:1][2:1].bf16.col, R72[4:2][2:1].f32
[NwwDs--]      DMA.LD.B64 R[234:235], SH[12:13], R1, _, _, _, _,  SH[12:14]
[N--D---]      FOP.ADD R42.h.bf16, C0, R99
[N--D---]      FOP.ADD R43.l.bf16, C0, R106
[N--D---]      FOP.ADD R43.h.bf16, C0, R107
[N--D---]      FOP.ADD R44.l.bf16, C0, R100
[N--D---]      FOP.ADD R44.h.bf16, C0, R101
[N--D---]      FOP.ADD R45.l.bf16, C0, R108
[N--D---]      FOP.ADD R45.h.bf16, C0, R109
[N--D---]      FOP.ADD R46.l.bf16, C0, R102
[N--D---] MMA.323216 R48[4:2][2:1].f32, R192[2:1][1:1].bf16.row, R209[1:1][2:1].bf16.col, R48[4:2][2:1].f32
[N--D---]      FOP.ADD R46.h.bf16, C0, R103
[N--D---]      FOP.ADD R47.l.bf16, C0, R110
[N-sD---]      FOP.ADD R47.h.bf16, C0, R111
[N-wD---]      LDMA.WR.B64.BST1 R[32:33], R6, X4, 128 , BURST_STRIDE=4
[N--D---]      LDMA.WR.B64.BST1 R[34:35], R6, X4, 256 , BURST_STRIDE=4
[N--D---]      LDMA.WR.B64.BST1 R[36:37], R6, X4, 640 , BURST_STRIDE=4
[N--D---] MMA.323216 R56[4:2][2:1].f32, R192[2:1][1:1].bf16.row, R211[1:1][2:1].bf16.col, R56[4:2][2:1].f32
[Nw-D-s-]       LDMA.RD.B32.BST4  R[217:220], R3, X4, 34944, BURST_STRIDE=512
[N--D---]      LDMA.WR.B64.BST1 R[38:39], R6, X4, 768 , BURST_STRIDE=4
[N--D---]      LDMA.WR.B64.BST1 R[40:41], R6, X4, 1152 , BURST_STRIDE=4
[N--D---]      LDMA.WR.B64.BST1 R[42:43], R6, X4, 1280 , BURST_STRIDE=4
[N--D---]      LDMA.WR.B64.BST1 R[44:45], R6, X4, 1664 , BURST_STRIDE=4
[N--D--o]      LDMA.WR.B64.BST1 R[46:47], R6, X4, 1792 , BURST_STRIDE=4
; SOFTMAX Reg2Sh END
[N--D---] MMA.323216 R64[4:2][2:1].f32, R194[2:1][1:1].bf16.row, R209[1:1][2:1].bf16.col, R64[4:2][2:1].f32
[Ns-D-w-] MMA.323216 R72[4:2][2:1].f32, R194[2:1][1:1].bf16.row, R211[1:1][2:1].bf16.col, R72[4:2][2:1].f32
[N--D---] MMA.323216 R48[4:2][2:1].f32, R196[2:1][1:1].bf16.row, R217[1:1][2:1].bf16.col, R48[4:2][2:1].f32
[NwwD---]       AP.BARRIER LM0
[N--Dw--]       CTRL.NOP
[N--D-s-]       LDMA.WR.B64.BST1 R[234:235], R4, X4, 32896 , BURST_STRIDE=16
[Ns-D-w-]       INT.ADD R1.s32, SH56.s32, R1.s32
[Nw-D---]       AP.BARRIER LM0
[N--D---] MMA.323216 R56[4:2][2:1].f32, R196[2:1][1:1].bf16.row, R219[1:1][2:1].bf16.col, R56[4:2][2:1].f32
[Nw-D-s-]       LDMA.RD.B32.BST4  R[209:212], R3, X4, 32896, BURST_STRIDE=512
[N--D---] MMA.323216 R64[4:2][2:1].f32, R198[2:1][1:1].bf16.row, R217[1:1][2:1].bf16.col, R64[4:2][2:1].f32
[Ns-D-w-] MMA.323216 R72[4:2][2:1].f32, R198[2:1][1:1].bf16.row, R219[1:1][2:1].bf16.col, R72[4:2][2:1].f32
[Nw-Ds--]      DMA.LD.B128 R[234:237] , SH[16:17], R2, _, _, _, _,  SH[16:18]
[N--D---] MMA.323216 R48[4:2][2:1].f32, R200[2:1][1:1].bf16.row, R209[1:1][2:1].bf16.col, R48[4:2][2:1].f32
[N--D---] MMA.323216 R56[4:2][2:1].f32, R200[2:1][1:1].bf16.row, R211[1:1][2:1].bf16.col, R56[4:2][2:1].f32
[Nw-D-s-]       LDMA.RD.B32.BST4  R[217:220], R3, X4, 34944, BURST_STRIDE=512
[N--D---] MMA.323216 R64[4:2][2:1].f32, R202[2:1][1:1].bf16.row, R209[1:1][2:1].bf16.col, R64[4:2][2:1].f32
[Ns-D-w-] MMA.323216 R72[4:2][2:1].f32, R202[2:1][1:1].bf16.row, R211[1:1][2:1].bf16.col, R72[4:2][2:1].f32
[N--D---] MMA.323216 R48[4:2][2:1].f32, R204[2:1][1:1].bf16.row, R217[1:1][2:1].bf16.col, R48[4:2][2:1].f32
[NwwD---]       AP.BARRIER LM0
[N--Dw--]       CTRL.NOP
[N--D-s-]       LDMA.WR.B128.BST1 R[234:237], R5, X4, 32896 , BURST_STRIDE=16
[Ns-D-w-]       INT.ADD R2.s32, SH57.s32, R2.s32
[Nw-D---]       AP.BARRIER LM0
[N--D---] MMA.323216 R56[4:2][2:1].f32, R204[2:1][1:1].bf16.row, R219[1:1][2:1].bf16.col, R56[4:2][2:1].f32
[N--D-s-]       LDMA.RD.B32.BST4  R[209:212], R3, X4, 32896, BURST_STRIDE=512
[N--D-s-]       LDMA.RD.B32.BST4  R[213:216], R3, X4, 34944, BURST_STRIDE=512
[N--D---] MMA.323216 R64[4:2][2:1].f32, R206[2:1][1:1].bf16.row, R217[1:1][2:1].bf16.col, R64[4:2][2:1].f32
[Nw-D-w-]       CTRL.NOP
[Nw-D-s-]      LDMA.RD.B32.BST4  R[226:229], R7, X4, 128, BURST_STRIDE=512
[Ns-D-w-] MMA.323216 R72[4:2][2:1].f32, R206[2:1][1:1].bf16.row, R219[1:1][2:1].bf16.col, R72[4:2][2:1].f32
[Nw-Ds--]      DMA.LD.B128 R[234:237], SH[16:17], R2, _, _, _, _,  SH[16:18]
[N-wD---]      FOP.ADD R30, R80, R81
[N--D---]      FOP.ADD R31, R82, R83
[N--D---]      FOP.ADD R32, R84, R85
[N--D---]      FOP.ADD R33, R86, R87
[N--D---]      FOP.ADD R34, R96, R97
[N--D---]      FOP.ADD R35, R98, R99
[N--D---]      FOP.ADD R36, R100, R101
[N--D---]      FOP.ADD R37, R102, R103
[N--D---] MMA.323216 R112[4:2][2:1].f32, R226[2:1][1:1].bf16.row, R209[1:1][2:1].bf16.row, R112[4:2][2:1].f32
[N--D---]      FOP.ADD R30, R30, R88
[N--D---]      FOP.ADD R31, R31, R90
[N--D---]      FOP.ADD R32, R32, R92
[N--D---]      FOP.ADD R33, R33, R94
[N--D---] MMA.323216 R120[4:2][2:1].f32, R226[2:1][1:1].bf16.row, R211[1:1][2:1].bf16.row, R120[4:2][2:1].f32
[Nw-D-s-]       LDMA.RD.B32.BST4  R[230:233], R7, X4, 8320, BURST_STRIDE=512
[N--D-s-]       LDMA.RD.B32.BST4  R[217:220], R3, X4, 36992, BURST_STRIDE=512
[N--D-s-]       LDMA.RD.B32.BST4  R[221:224], R3, X4, 39040, BURST_STRIDE=512
[N--D---]      FOP.ADD R34, R34, R104
[N--D---]      FOP.ADD R35, R35, R106
[N--D---]      FOP.ADD R36, R36, R108
[N--D---]      FOP.ADD R37, R37, R110
[N--D---]      FOP.ADD R30, R30, R89
[N--D---]      FOP.ADD R31, R31, R91
[N--D---]      FOP.ADD R32, R32, R93
[N--D---]      FOP.ADD R33, R33, R95
[N--D---] MMA.323216 R128[4:2][2:1].f32, R226[2:1][1:1].bf16.row, R213[1:1][2:1].bf16.row, R128[4:2][2:1].f32
[N--D---]      FOP.ADD R34, R34, R105
[N--D---]      FOP.ADD R35, R35, R107
[N--D---]      FOP.ADD R36, R36, R109
[N-sD---]      FOP.ADD R37, R37, R111
[N-wD-s-]      LDMA.WR.B32.BST4 R[30:33], R8, X4, 41088 , BURST_STRIDE=512
[N--D-s-]      LDMA.WR.B32.BST4 R[34:37], R8, X4, 43136 , BURST_STRIDE=512
[NwwD-w-]    CTRL.NOP
[NwwD---]    AP.BARRIER LM0
[N--D---] MMA.323216 R136[4:2][2:1].f32, R226[2:1][1:1].bf16.row, R215[1:1][2:1].bf16.row, R136[4:2][2:1].f32
[Nw-D---]      LDMA.RD.B128.BST1 R[30:33], R9, X4, 41088 , BURST_STRIDE=16
[N-sD---]      LDMA.RD.B128.BST1 R[34:37], R9, X4, 41120 , BURST_STRIDE=16
[N-wD---]      FOP.ADD R30, R30, R31
[N--D---]      FOP.ADD R32, R32, R33
[N--D---]      FOP.ADD R34, R34, R35
[N--D---]      FOP.ADD R36, R36, R37
[N--D---]      FOP.ADD R30, R30, R32
[N--D---]      FOP.ADD R34, R34, R36
[N--D---] MMA.323216 R144[4:2][2:1].f32, R228[2:1][1:1].bf16.row, R209[1:1][2:1].bf16.row, R144[4:2][2:1].f32
[N--D---]      FOP.ADD R30, R30, R34
[N--D---]      FOP.SWPX R31, R30
[N--D---]      FOP.ADD R30, R30, R31
[N--D---]      FOP.FMA R225, R225, R18, R30
[N-sD---]      FOP.MOV I0, R19
[N-sD---]       INT.BYP.TE p0, SL0.s32, C33.s32
[NwwD---]  P0  FOP.FMA R80, SH50, R48, R238
[N--D---]  P0  FOP.FMA R81, SH50, R49, R239
[N--D---] MMA.323216 R152[4:2][2:1].f32, R228[2:1][1:1].bf16.row, R211[1:1][2:1].bf16.row, R152[4:2][2:1].f32
[NwwD---]  !P0  FOP.FMA R80, SH50, R48, C0
[N--D---]  !P0  FOP.FMA R81, SH50, R49, C0
[N--D---]  P0  FOP.FMA R82, SH50, R50, R238
[N--D---]  P0  FOP.FMA R83, SH50, R51, R239
[N--D---]  !P0  FOP.FMA R82, SH50, R50, C0
[N--D---]  !P0  FOP.FMA R83, SH50, R51, C0
[N--D---]  P0  FOP.FMA R84, SH50, R52, R238
[N--D---]  P0  FOP.FMA R85, SH50, R53, R239
[N--D---] MMA.323216 R160[4:2][2:1].f32, R228[2:1][1:1].bf16.row, R213[1:1][2:1].bf16.row, R160[4:2][2:1].f32
[N--D---]  !P0  FOP.FMA R84, SH50, R52, C0
[N--D---]  !P0  FOP.FMA R85, SH50, R53, C0
[N--D---]  P0  FOP.FMA R86, SH50, R54, R238
[N--D---]  P0  FOP.FMA R87, SH50, R55, R239
[N--D---]  !P0  FOP.FMA R86, SH50, R54, C0
[N--D---]  !P0  FOP.FMA R87, SH50, R55, C0
[N--D---]  P0  FOP.FMA R88, SH50, R56, R240
[N--D---]  P0  FOP.FMA R89, SH50, R57, R241
[Ns-D-w-] MMA.323216 R168[4:2][2:1].f32, R228[2:1][1:1].bf16.row, R215[1:1][2:1].bf16.row, R168[4:2][2:1].f32
[N--D---]  !P0  FOP.FMA R88, SH50, R56, C0
[N--D---]  !P0  FOP.FMA R89, SH50, R57, C0
[N--D---]  P0  FOP.FMA R90, SH50, R58, R240
[N--D---]  P0  FOP.FMA R91, SH50, R59, R241
[N--D---]  !P0  FOP.FMA R90, SH50, R58, C0
[N--D---]  !P0  FOP.FMA R91, SH50, R59, C0
[N--D---]  P0  FOP.FMA R92, SH50, R60, R240
[N--D---]  P0  FOP.FMA R93, SH50, R61, R241
[N--D---] MMA.323216 R112[4:2][2:1].f32, R230[2:1][1:1].bf16.row, R217[1:1][2:1].bf16.row, R112[4:2][2:1].f32
[NwwD---]       AP.BARRIER LM0
[N--Dw--]       CTRL.NOP
[N--D-s-]       LDMA.WR.B128.BST1 R[234:237], R5, X4, 32896 , BURST_STRIDE=16
[Ns-D-w-]       INT.ADD R2.s32, SH57.s32, R2.s32
[Nw-D---]       AP.BARRIER LM0
[N--D---]  !P0  FOP.FMA R92, SH50, R60, C0
[N--D---]  !P0  FOP.FMA R93, SH50, R61, C0
[N--D---]  P0  FOP.FMA R94, SH50, R62, R240
[N--D---]  P0  FOP.FMA R95, SH50, R63, R241
[N--D---] MMA.323216 R120[4:2][2:1].f32, R230[2:1][1:1].bf16.row, R219[1:1][2:1].bf16.row, R120[4:2][2:1].f32
[Nw-D-s-]       LDMA.RD.B32.BST4  R[226:229], R7, X4, 16512, BURST_STRIDE=512
[N--D-s-]       LDMA.RD.B32.BST4  R[209:212], R3, X4, 32896, BURST_STRIDE=512
[N--D-s-]       LDMA.RD.B32.BST4  R[213:216], R3, X4, 34944, BURST_STRIDE=512
[N--D---]  !P0  FOP.FMA R94, SH50, R62, C0
[N--D---]  !P0  FOP.FMA R95, SH50, R63, C0
[N--D---]  P0  FOP.FMA R96, SH50, R64, R238
[N--D---]  P0  FOP.FMA R97, SH50, R65, R239
[N--D---]  !P0  FOP.FMA R96, SH50, R64, C0
[N--D---]  !P0  FOP.FMA R97, SH50, R65, C0
[N--D---]  P0  FOP.FMA R98, SH50, R66, R238
[N--D---]  P0  FOP.FMA R99, SH50, R67, R239
[N--D---] MMA.323216 R128[4:2][2:1].f32, R230[2:1][1:1].bf16.row, R221[1:1][2:1].bf16.row, R128[4:2][2:1].f32
[N--D---]  !P0  FOP.FMA R98, SH50, R66, C0
[N--D---]  !P0  FOP.FMA R99, SH50, R67, C0
[N--D---]  P0  FOP.FMA R100, SH50, R68, R238
[N--D---]  P0  FOP.FMA R101, SH50, R69, R239
[N--D---]  !P0  FOP.FMA R100, SH50, R68, C0
[N--D---]  !P0  FOP.FMA R101, SH50, R69, C0
[N--D---]  P0  FOP.FMA R102, SH50, R70, R238
[N--D---]  P0  FOP.FMA R103, SH50, R71, R239
[N--D---] MMA.323216 R136[4:2][2:1].f32, R230[2:1][1:1].bf16.row, R223[1:1][2:1].bf16.row, R136[4:2][2:1].f32
[N--D---]  !P0  FOP.FMA R102, SH50, R70, C0
[N--D---]  !P0  FOP.FMA R103, SH50, R71, C0
[N--D---]  P0  FOP.FMA R104, SH50, R72, R240
[N--D---]  P0  FOP.FMA R105, SH50, R73, R241
[N--D---]  !P0  FOP.FMA R104, SH50, R72, C0
[N--D---]  !P0  FOP.FMA R105, SH50, R73, C0
[N--D---]  P0  FOP.FMA R106, SH50, R74, R240
[N--D---]  P0  FOP.FMA R107, SH50, R75, R241
[N--D---] MMA.323216 R144[4:2][2:1].f32, R232[2:1][1:1].bf16.row, R217[1:1][2:1].bf16.row, R144[4:2][2:1].f32
[N--D---]  !P0  FOP.FMA R106, SH50, R74, C0
[N--D---]  !P0  FOP.FMA R107, SH50, R75, C0
[N--D---]  P0  FOP.FMA R108, SH50, R76, R240
[N--D---]  P0  FOP.FMA R109, SH50, R77, R241
[N--D---]  !P0  FOP.FMA R108, SH50, R76, C0
[N--D---]  !P0  FOP.FMA R109, SH50, R77, C0
[N--D---]  P0  FOP.FMA R110, SH50, R78, R240
[N-sD--w]  P0  FOP.FMA R111, SH50, R79, R241
[N--D---] MMA.323216 R152[4:2][2:1].f32, R232[2:1][1:1].bf16.row, R219[1:1][2:1].bf16.row, R152[4:2][2:1].f32
[N--D---]  !P0  FOP.FMA R110, SH50, R78, C0
[N-sD---]  !P0  FOP.FMA R111, SH50, R79, C0
; SOFTMAX MAX REDUCE BEG
[NwwD---]      FOP.CMP_MOVC.MAX.TST R30, R80, R81
[N--D---]      FOP.CMP_MOVC.MAX.TST R31, R82, R83
[N--D---]      FOP.CMP_MOVC.MAX.TST R32, R84, R85
[N--D---]      FOP.CMP_MOVC.MAX.TST R33, R86, R87
[N--D---]      FOP.CMP_MOVC.MAX.TST R34, R96, R97
[N--D---] MMA.323216 R160[4:2][2:1].f32, R232[2:1][1:1].bf16.row, R221[1:1][2:1].bf16.row, R160[4:2][2:1].f32
[N--D---]      FOP.CMP_MOVC.MAX.TST R35, R98, R99
[N--D---]      FOP.CMP_MOVC.MAX.TST R36, R100, R101
[N--D---]      FOP.CMP_MOVC.MAX.TST R37, R102, R103
[N--D---]      FOP.CMP_MOVC.MAX.TST R30, R30, R88
[N--D---]      FOP.CMP_MOVC.MAX.TST R31, R31, R90
[N--D---]      FOP.CMP_MOVC.MAX.TST R32, R32, R92
[N--D---]      FOP.CMP_MOVC.MAX.TST R33, R33, R94
[N--D---]      FOP.CMP_MOVC.MAX.TST R34, R34, R104
[Ns-D-w-] MMA.323216 R168[4:2][2:1].f32, R232[2:1][1:1].bf16.row, R223[1:1][2:1].bf16.row, R168[4:2][2:1].f32
[Nw-Ds--]      DMA.LD.B64 R[234:235], SH[12:13], R1, _, _, _, _,  SH[12:14]
[N--D---]      FOP.CMP_MOVC.MAX.TST R35, R35, R106
[N--D---]      FOP.CMP_MOVC.MAX.TST R36, R36, R108
[N--D---]      FOP.CMP_MOVC.MAX.TST R37, R37, R110
[N--D---]      FOP.CMP_MOVC.MAX.TST R30, R30, R89
[N--D---]      FOP.CMP_MOVC.MAX.TST R31, R31, R91
[N--D---]      FOP.CMP_MOVC.MAX.TST R32, R32, R93
[N--D---]      FOP.CMP_MOVC.MAX.TST R33, R33, R95
[N--D---]      FOP.CMP_MOVC.MAX.TST R34, R34, R105
[N--D---] MMA.323216 R112[4:2][2:1].f32, R226[2:1][1:1].bf16.row, R209[1:1][2:1].bf16.row, R112[4:2][2:1].f32
[N--D---]      FOP.CMP_MOVC.MAX.TST R35, R35, R107
[N--D---]      FOP.CMP_MOVC.MAX.TST R36, R36, R109
[N-sD---]      FOP.CMP_MOVC.MAX.TST R37, R37, R111
[N-wD-s-]      LDMA.WR.B32.BST4 R[30:33], R8, X4, 41088 , BURST_STRIDE=512
[N--D---] MMA.323216 R120[4:2][2:1].f32, R226[2:1][1:1].bf16.row, R211[1:1][2:1].bf16.row, R120[4:2][2:1].f32
[Nw-D-s-]       LDMA.RD.B32.BST4  R[230:233], R7, X4, 24704, BURST_STRIDE=512
[N--D-s-]       LDMA.RD.B32.BST4  R[217:220], R3, X4, 36992, BURST_STRIDE=512
[N--D-s-]       LDMA.RD.B32.BST4  R[221:224], R3, X4, 39040, BURST_STRIDE=512
[N--D-s-]      LDMA.WR.B32.BST4 R[34:37], R8, X4, 43136 , BURST_STRIDE=512
[NwwD-w-]    CTRL.NOP
[Nw-D---]      LDMA.RD.B128.BST1 R[30:33], R9, X4, 41088 , BURST_STRIDE=16
[N-sD---]      LDMA.RD.B128.BST1 R[34:37], R9, X4, 41120 , BURST_STRIDE=16
[N-wD---]      FOP.CMP_MOVC.MAX.TST R30, R30, R31
[N--D---]      FOP.CMP_MOVC.MAX.TST R32, R32, R33
[N--D---]      FOP.CMP_MOVC.MAX.TST R34, R34, R35
[N--D---]      FOP.CMP_MOVC.MAX.TST R36, R36, R37
[N--D---] MMA.323216 R128[4:2][2:1].f32, R226[2:1][1:1].bf16.row, R213[1:1][2:1].bf16.row, R128[4:2][2:1].f32
[N--D---]      FOP.CMP_MOVC.MAX.TST R30, R30, R32
[N--D---]      FOP.CMP_MOVC.MAX.TST R34, R34, R36
[N--D---]      FOP.CMP_MOVC.MAX.TST R30, R30, R34
[N--D---]      FOP.SWPX R31, R30
[N--D---]      FOP.CMP_MOVC.MAX.TST R30, R30, R31
[N--D---]      FOP.CMP_MOVC.MAX.TST R19, R30, I0
[N-sD---]      FOP.ADD R18, I0, R19.neg
[N-wD--s]      AF32.EXP2 R18, R18
[N--D---] MMA.323216 R136[4:2][2:1].f32, R226[2:1][1:1].bf16.row, R215[1:1][2:1].bf16.row, R136[4:2][2:1].f32
[N-wD--w]      SHUFFLE.IDX R30, R19, R10, SH81
[N--D---]      SHUFFLE.IDX R242, R18, R10, SH81
[N--D---]      SHUFFLE.IDX R31, R19, R11, SH81
[N--D---]      SHUFFLE.IDX R243, R18, R11, SH81
[N--D---]      SHUFFLE.IDX R32, R19, R12, SH81
[N--D---]      SHUFFLE.IDX R208, R18, R12, SH81
[N--D---]      SHUFFLE.IDX R33, R19, R13, SH81
[N--D---]      SHUFFLE.IDX AR1, R18, R13, SH81
[N--D---] MMA.323216 R144[4:2][2:1].f32, R228[2:1][1:1].bf16.row, R209[1:1][2:1].bf16.row, R144[4:2][2:1].f32
[N--D---]      SHUFFLE.IDX R34, R19, R14, SH81
[N--D---]      SHUFFLE.IDX AR2, R18, R14, SH81
[N--D---]      SHUFFLE.IDX R35, R19, R15, SH81
[N--D---]      SHUFFLE.IDX AR3, R18, R15, SH81
[N--D---]      SHUFFLE.IDX R36, R19, R16, SH81
[N--D---]      SHUFFLE.IDX AR4, R18, R16, SH81
[N--D---]      SHUFFLE.IDX R37, R19, R17, SH81
[N-sD---]      SHUFFLE.IDX AR5, R18, R17, SH81
[N--D---] MMA.323216 R152[4:2][2:1].f32, R228[2:1][1:1].bf16.row, R211[1:1][2:1].bf16.row, R152[4:2][2:1].f32
[N-wD---]      FOP.ADD R80, R80, R30.neg
[N--D---]      FOP.ADD R81, R81, R30.neg
[N--D---]      FOP.ADD R88, R88, R30.neg
[N--D---]      FOP.ADD R89, R89, R30.neg
[N--D---]      FOP.ADD R82, R82, R31.neg
[N--D---]      FOP.ADD R83, R83, R31.neg
[N--D---]      FOP.ADD R90, R90, R31.neg
[N--D---]      FOP.ADD R91, R91, R31.neg
[N--D---] MMA.323216 R160[4:2][2:1].f32, R228[2:1][1:1].bf16.row, R213[1:1][2:1].bf16.row, R160[4:2][2:1].f32
[N--D---]      FOP.ADD R84, R84, R32.neg
[N--D---]      FOP.ADD R85, R85, R32.neg
[N--D---]      FOP.ADD R92, R92, R32.neg
[N--D---]      FOP.ADD R93, R93, R32.neg
[N--D---]      FOP.ADD R86, R86, R33.neg
[N--D---]      FOP.ADD R87, R87, R33.neg
[N--D---]      FOP.ADD R94, R94, R33.neg
[N--D---]      FOP.ADD R95, R95, R33.neg
[Ns-D-w-] MMA.323216 R168[4:2][2:1].f32, R228[2:1][1:1].bf16.row, R215[1:1][2:1].bf16.row, R168[4:2][2:1].f32
[N--D---]      FOP.ADD R96, R96, R34.neg
[N--D---]      FOP.ADD R97, R97, R34.neg
[N--D---]      FOP.ADD R104, R104, R34.neg
[N--D---]      FOP.ADD R105, R105, R34.neg
[N--D---]      FOP.ADD R98, R98, R35.neg
[N--D---]      FOP.ADD R99, R99, R35.neg
[N--D---]      FOP.ADD R106, R106, R35.neg
[N--D---]      FOP.ADD R107, R107, R35.neg
[N--D---] MMA.323216 R112[4:2][2:1].f32, R230[2:1][1:1].bf16.row, R217[1:1][2:1].bf16.row, R112[4:2][2:1].f32
[NwwD---]       AP.BARRIER LM0
[N--Dw--]       CTRL.NOP
[N--D-s-]       LDMA.WR.B64.BST1 R[234:235], R4, X4, 32896 , BURST_STRIDE=16
[Ns-D-w-]       INT.ADD R1.s32, SH77.s32, R1.s32
[Nw-D---]       AP.BARRIER LM0
[N--D---]      FOP.ADD R100, R100, R36.neg
[N--D---]      FOP.ADD R101, R101, R36.neg
[N--D---]      FOP.ADD R108, R108, R36.neg
[N--D---]      FOP.ADD R109, R109, R36.neg
[N--D---] MMA.323216 R120[4:2][2:1].f32, R230[2:1][1:1].bf16.row, R219[1:1][2:1].bf16.row, R120[4:2][2:1].f32
[Nw-D-s-]       LDMA.RD.B32.BST4  R[209:212], R3, X4, 32896, BURST_STRIDE=512
[N--D---]      FOP.ADD R102, R102, R37.neg
[N--D---]      FOP.ADD R103, R103, R37.neg
[N--D---]      FOP.ADD R110, R110, R37.neg
[N-sD---]      FOP.ADD R111, R111, R37.neg
; SOFTMAX MAX REDUCE END
[N--D---] MMA.323216 R128[4:2][2:1].f32, R230[2:1][1:1].bf16.row, R221[1:1][2:1].bf16.row, R128[4:2][2:1].f32
[N--D---] MMA.323216 R136[4:2][2:1].f32, R230[2:1][1:1].bf16.row, R223[1:1][2:1].bf16.row, R136[4:2][2:1].f32
[N--D---] MMA.323216 R144[4:2][2:1].f32, R232[2:1][1:1].bf16.row, R217[1:1][2:1].bf16.row, R144[4:2][2:1].f32
[N--D---] MMA.323216 R152[4:2][2:1].f32, R232[2:1][1:1].bf16.row, R219[1:1][2:1].bf16.row, R152[4:2][2:1].f32
[N--D---] MMA.323216 R160[4:2][2:1].f32, R232[2:1][1:1].bf16.row, R221[1:1][2:1].bf16.row, R160[4:2][2:1].f32
[Ns-D-w-] MMA.323216 R168[4:2][2:1].f32, R232[2:1][1:1].bf16.row, R223[1:1][2:1].bf16.row, R168[4:2][2:1].f32
[NwsD---]      INT.ADD SL1.s32, C33.neg.s32, SL1.s32
[NswD---]      INT.ADD SL0.s32, C33.neg.s32, SL0.s32
[NswD---]       INT.BYP.TG p0, SL1.s32, C0.s32
[Nw-D---]     CTRL.BR.ANYP0T MAIN_LOOP
LAST_LOOP:
; SOFTMAX UPDATE ACC BEG
[N-wD---]      AF32.EXP2 R80, R80
[NwwD---]      FOP.MUL R112, R112, R242
[N--D---]      AF32.EXP2 R81, R81
[N--D---]      FOP.MUL R113, R113, R242
[N--D---]      AF32.EXP2 R88, R88
[N--D---]      FOP.MUL R120, R120, R242
[N--D---]      AF32.EXP2 R89, R89
[N--D---]      FOP.MUL R121, R121, R242
[N--D---]      FOP.MUL R128, R128, R242
[N--D---]      FOP.MUL R129, R129, R242
[N--D---]      FOP.MUL R136, R136, R242
[N--D---]      FOP.MUL R137, R137, R242
[N--D---]      AF32.EXP2 R82, R82
[N--D---]      FOP.MUL R114, R114, R243
[N--D---]      AF32.EXP2 R83, R83
[N--D---]      FOP.MUL R115, R115, R243
[N--D---]      AF32.EXP2 R90, R90
[N--D---]      FOP.MUL R122, R122, R243
[N--D---]      AF32.EXP2 R91, R91
[N--D---]      FOP.MUL R123, R123, R243
[N--D---]      FOP.MUL R130, R130, R243
[N--D---]      FOP.MUL R131, R131, R243
[N--D---]      FOP.MUL R138, R138, R243
[N--D---]      FOP.MUL R139, R139, R243
[N--D---]      AF32.EXP2 R84, R84
[N--D---]      FOP.MUL R116, R116, R208
[N--D---]      AF32.EXP2 R85, R85
[N--D---]      FOP.MUL R117, R117, R208
[N--D---]      AF32.EXP2 R92, R92
[N--D---]      FOP.MUL R124, R124, R208
[N--D---]      AF32.EXP2 R93, R93
[N--D---]      FOP.MUL R125, R125, R208
[N--D---]      FOP.MUL R132, R132, R208
[N--D---]      FOP.MUL R133, R133, R208
[N--D---]      FOP.MUL R140, R140, R208
[N--D---]      FOP.MUL R141, R141, R208
[N--D---]      AF32.EXP2 R86, R86
[N--D---]      FOP.MUL R118, R118, AR1
[N--D---]      AF32.EXP2 R87, R87
[N--D---]      FOP.MUL R119, R119, AR1
[N--D---]      AF32.EXP2 R94, R94
[N--D---]      FOP.MUL R126, R126, AR1
[N--D---]      AF32.EXP2 R95, R95
[N--D---]      FOP.MUL R127, R127, AR1
[N--D---]      FOP.MUL R134, R134, AR1
[N--D---]      FOP.MUL R135, R135, AR1
[N--D---]      FOP.MUL R142, R142, AR1
[N--D---]      FOP.MUL R143, R143, AR1
[N--D---]      AF32.EXP2 R96, R96
[N--D---]      FOP.MUL R144, R144, AR2
[N--D---]      AF32.EXP2 R97, R97
[N--D---]      FOP.MUL R145, R145, AR2
[N--D---]      AF32.EXP2 R104, R104
[N--D---]      FOP.MUL R152, R152, AR2
[N--D---]      AF32.EXP2 R105, R105
[N--D---]      FOP.MUL R153, R153, AR2
[N--D---]      FOP.MUL R160, R160, AR2
[N--D---]      FOP.MUL R161, R161, AR2
[N--D---]      FOP.MUL R168, R168, AR2
[N--D---]      FOP.MUL R169, R169, AR2
[N--D---]      AF32.EXP2 R98, R98
[N--D---]      FOP.MUL R146, R146, AR3
[N--D---]      AF32.EXP2 R99, R99
[N--D---]      FOP.MUL R147, R147, AR3
[N--D---]      AF32.EXP2 R106, R106
[N--D---]      FOP.MUL R154, R154, AR3
[N--D---]      AF32.EXP2 R107, R107
[N--D---]      FOP.MUL R155, R155, AR3
[N--D---]      FOP.MUL R162, R162, AR3
[N--D---]      FOP.MUL R163, R163, AR3
[N--D---]      FOP.MUL R170, R170, AR3
[N--D---]      FOP.MUL R171, R171, AR3
[N--D---]      AF32.EXP2 R100, R100
[N--D---]      FOP.MUL R148, R148, AR4
[N--D---]      AF32.EXP2 R101, R101
[N--D---]      FOP.MUL R149, R149, AR4
[N--D---]      AF32.EXP2 R108, R108
[N--D---]      FOP.MUL R156, R156, AR4
[N--D---]      AF32.EXP2 R109, R109
[N--D---]      FOP.MUL R157, R157, AR4
[N--D---]      FOP.MUL R164, R164, AR4
[N--D---]      FOP.MUL R165, R165, AR4
[N--D---]      FOP.MUL R172, R172, AR4
[N--D---]      FOP.MUL R173, R173, AR4
[N--D---]      AF32.EXP2 R102, R102
[N--D---]      FOP.MUL R150, R150, AR5
[N--D---]      AF32.EXP2 R103, R103
[N--D---]      FOP.MUL R151, R151, AR5
[N--D---]      AF32.EXP2 R110, R110
[N--D---]      FOP.MUL R158, R158, AR5
[N--D--s]      AF32.EXP2 R111, R111
[N--D---]      FOP.MUL R159, R159, AR5
[N--D---]      FOP.MUL R166, R166, AR5
[N--D---]      FOP.MUL R167, R167, AR5
[N--D---]      FOP.MUL R174, R174, AR5
[N--D--w]      FOP.MUL R175, R175, AR5
; SOFTMAX UPDATE ACC END
; SOFTMAX Reg2Sh BEG
[N--D---]      FOP.ADD R32.l.bf16, C0, R80
[N--D---]      FOP.ADD R32.h.bf16, C0, R81
[N--D---]      FOP.ADD R33.l.bf16, C0, R88
[N--D---]      FOP.ADD R33.h.bf16, C0, R89
[N--D---]      FOP.ADD R34.l.bf16, C0, R82
[N--D---]      FOP.ADD R34.h.bf16, C0, R83
[N--D---]      FOP.ADD R35.l.bf16, C0, R90
[N--D---]      FOP.ADD R35.h.bf16, C0, R91
[N--D---]      FOP.ADD R36.l.bf16, C0, R84
[N--D---]      FOP.ADD R36.h.bf16, C0, R85
[N--D---]      FOP.ADD R37.l.bf16, C0, R92
[N--D---]      FOP.ADD R37.h.bf16, C0, R93
[N--D---]      FOP.ADD R38.l.bf16, C0, R86
[N--D---]      FOP.ADD R38.h.bf16, C0, R87
[N--D---]      FOP.ADD R39.l.bf16, C0, R94
[N--D---]      FOP.ADD R39.h.bf16, C0, R95
[N--D---]      FOP.ADD R40.l.bf16, C0, R96
[N--D---]      FOP.ADD R40.h.bf16, C0, R97
[N--D---]      FOP.ADD R41.l.bf16, C0, R104
[N--D---]      FOP.ADD R41.h.bf16, C0, R105
[N--D---]      FOP.ADD R42.l.bf16, C0, R98
[N--D---]      FOP.ADD R42.h.bf16, C0, R99
[N--D---]      FOP.ADD R43.l.bf16, C0, R106
[N--D---]      FOP.ADD R43.h.bf16, C0, R107
[N--D---]      FOP.ADD R44.l.bf16, C0, R100
[N--D---]      FOP.ADD R44.h.bf16, C0, R101
[N--D---]      FOP.ADD R45.l.bf16, C0, R108
[N--D---]      FOP.ADD R45.h.bf16, C0, R109
[N--D---]      FOP.ADD R46.l.bf16, C0, R102
[N--D---]      FOP.ADD R46.h.bf16, C0, R103
[N--D---]      FOP.ADD R47.l.bf16, C0, R110
[N-sD---]      FOP.ADD R47.h.bf16, C0, R111
[N-wD---]      LDMA.WR.B64.BST1 R[32:33], R6, X4, 128 , BURST_STRIDE=4
[N--D---]      LDMA.WR.B64.BST1 R[34:35], R6, X4, 256 , BURST_STRIDE=4
[N--D---]      LDMA.WR.B64.BST1 R[36:37], R6, X4, 640 , BURST_STRIDE=4
[N--D---]      LDMA.WR.B64.BST1 R[38:39], R6, X4, 768 , BURST_STRIDE=4
[N--D---]      LDMA.WR.B64.BST1 R[40:41], R6, X4, 1152 , BURST_STRIDE=4
[N--D---]      LDMA.WR.B64.BST1 R[42:43], R6, X4, 1280 , BURST_STRIDE=4
[N--D---]      LDMA.WR.B64.BST1 R[44:45], R6, X4, 1664 , BURST_STRIDE=4
[N--D--o]      LDMA.WR.B64.BST1 R[46:47], R6, X4, 1792 , BURST_STRIDE=4
; SOFTMAX Reg2Sh END
[N-wD---]      FOP.ADD R30, R80, R81
[N--D---]      FOP.ADD R31, R82, R83
[N--D---]      FOP.ADD R32, R84, R85
[N--D---]      FOP.ADD R33, R86, R87
[N--D---]      FOP.ADD R34, R96, R97
[N--D---]      FOP.ADD R35, R98, R99
[N--D---]      FOP.ADD R36, R100, R101
[N--D---]      FOP.ADD R37, R102, R103
[N--D---]      FOP.ADD R30, R30, R88
[N--D---]      FOP.ADD R31, R31, R90
[N--D---]      FOP.ADD R32, R32, R92
[N--D---]      FOP.ADD R33, R33, R94
[N--D---]      FOP.ADD R34, R34, R104
[N--D---]      FOP.ADD R35, R35, R106
[N--D---]      FOP.ADD R36, R36, R108
[N--D---]      FOP.ADD R37, R37, R110
[N--D---]      FOP.ADD R30, R30, R89
[N--D---]      FOP.ADD R31, R31, R91
[N--D---]      FOP.ADD R32, R32, R93
[N--D---]      FOP.ADD R33, R33, R95
[N--D---]      FOP.ADD R34, R34, R105
[N--D---]      FOP.ADD R35, R35, R107
[N--D---]      FOP.ADD R36, R36, R109
[N-sD---]      FOP.ADD R37, R37, R111
[N-wD-s-]      LDMA.WR.B32.BST4 R[30:33], R8, X4, 41088 , BURST_STRIDE=512
[N--D-s-]      LDMA.WR.B32.BST4 R[34:37], R8, X4, 43136 , BURST_STRIDE=512
[NwwD-w-]    CTRL.NOP
[NwwD---]    AP.BARRIER LM0
[Nw-D---]      LDMA.RD.B128.BST1 R[30:33], R9, X4, 41088 , BURST_STRIDE=16
[N-sD---]      LDMA.RD.B128.BST1 R[34:37], R9, X4, 41120 , BURST_STRIDE=16
[N-wD---]      FOP.ADD R30, R30, R31
[N--D---]      FOP.ADD R32, R32, R33
[N--D---]      FOP.ADD R34, R34, R35
[N--D---]      FOP.ADD R36, R36, R37
[N--D---]      FOP.ADD R30, R30, R32
[N--D---]      FOP.ADD R34, R34, R36
[N--D---]      FOP.ADD R30, R30, R34
[N--D---]      FOP.SWPX R31, R30
[N--D---]      FOP.ADD R30, R30, R31
[N--D---]      FOP.FMA R225, R225, R18, R30
[N-sD---]      FOP.MOV I0, R19
[N-sD---]      AF32.RCP R20, R225
[N-wD---]      SHUFFLE.IDX R30, R20, R10, SH81
[N--D---]      SHUFFLE.IDX R31, R20, R11, SH81
[N--D---]      SHUFFLE.IDX R32, R20, R12, SH81
[N--D---]      SHUFFLE.IDX R33, R20, R13, SH81
[N--D---]      SHUFFLE.IDX R34, R20, R14, SH81
[N--D---]      SHUFFLE.IDX R35, R20, R15, SH81
[N--D---]      SHUFFLE.IDX R36, R20, R16, SH81
[N-sD---]      SHUFFLE.IDX R37, R20, R17, SH81
[NwwDo--]      DMA.LD.B128 R[234:237] , SH[16:17], R2, _, _, _, _,  SH[16:18]
[NwwD-o-]      LDMA.WR.B128.BST1 R[234:237], R5, X4, 32896 , BURST_STRIDE=16
[NwsD---]      INT.ADD R2.s32, SH57.s32, R2.s32
[NwwD---]      AP.BARRIER LM0
[N-wD-s-]      LDMA.RD.B32.BST4  R[226:229], R7, X4, 128, BURST_STRIDE=512
[N-wD-s-]      LDMA.RD.B32.BST4  R[209:212], R3, X4, 32896, BURST_STRIDE=512
[N-wD-o-]      LDMA.RD.B32.BST4  R[213:216], R3, X4, 34944, BURST_STRIDE=512
[NwwDo--]      DMA.LD.B128 R[234:237], SH[16:17], R2, _, _, _, _,  SH[16:18]
[N--D---] MMA.323216 R112[4:2][2:1].f32, R226[2:1][1:1].bf16.row, R209[1:1][2:1].bf16.row, R112[4:2][2:1].f32
[N--D---] MMA.323216 R120[4:2][2:1].f32, R226[2:1][1:1].bf16.row, R211[1:1][2:1].bf16.row, R120[4:2][2:1].f32
[Nw-D-s-]       LDMA.RD.B32.BST4  R[230:233], R7, X4, 8320, BURST_STRIDE=512
[N--D-s-]       LDMA.RD.B32.BST4  R[217:220], R3, X4, 36992, BURST_STRIDE=512
[N--D-s-]       LDMA.RD.B32.BST4  R[221:224], R3, X4, 39040, BURST_STRIDE=512
[N--D---] MMA.323216 R128[4:2][2:1].f32, R226[2:1][1:1].bf16.row, R213[1:1][2:1].bf16.row, R128[4:2][2:1].f32
[N--D---] MMA.323216 R136[4:2][2:1].f32, R226[2:1][1:1].bf16.row, R215[1:1][2:1].bf16.row, R136[4:2][2:1].f32
[N--D---] MMA.323216 R144[4:2][2:1].f32, R228[2:1][1:1].bf16.row, R209[1:1][2:1].bf16.row, R144[4:2][2:1].f32
[N--D---] MMA.323216 R152[4:2][2:1].f32, R228[2:1][1:1].bf16.row, R211[1:1][2:1].bf16.row, R152[4:2][2:1].f32
[N--D---] MMA.323216 R160[4:2][2:1].f32, R228[2:1][1:1].bf16.row, R213[1:1][2:1].bf16.row, R160[4:2][2:1].f32
[Ns-D-w-] MMA.323216 R168[4:2][2:1].f32, R228[2:1][1:1].bf16.row, R215[1:1][2:1].bf16.row, R168[4:2][2:1].f32
[N--D---] MMA.323216 R112[4:2][2:1].f32, R230[2:1][1:1].bf16.row, R217[1:1][2:1].bf16.row, R112[4:2][2:1].f32
[NwwD---]       AP.BARRIER LM0
[NwwD-o-]       LDMA.WR.B128.BST1 R[234:237], R5, X4, 32896 , BURST_STRIDE=16
[NwwD---]       AP.BARRIER LM0
[N--D---] MMA.323216 R120[4:2][2:1].f32, R230[2:1][1:1].bf16.row, R219[1:1][2:1].bf16.row, R120[4:2][2:1].f32
[Nw-D-s-]       LDMA.RD.B32.BST4  R[226:229], R7, X4, 16512, BURST_STRIDE=512
[N--D-s-]       LDMA.RD.B32.BST4  R[209:212], R3, X4, 32896, BURST_STRIDE=512
[N--D-s-]       LDMA.RD.B32.BST4  R[213:216], R3, X4, 34944, BURST_STRIDE=512
[N--D---] MMA.323216 R128[4:2][2:1].f32, R230[2:1][1:1].bf16.row, R221[1:1][2:1].bf16.row, R128[4:2][2:1].f32
[N--D---] MMA.323216 R136[4:2][2:1].f32, R230[2:1][1:1].bf16.row, R223[1:1][2:1].bf16.row, R136[4:2][2:1].f32
[N--D---] MMA.323216 R144[4:2][2:1].f32, R232[2:1][1:1].bf16.row, R217[1:1][2:1].bf16.row, R144[4:2][2:1].f32
[N--D---] MMA.323216 R152[4:2][2:1].f32, R232[2:1][1:1].bf16.row, R219[1:1][2:1].bf16.row, R152[4:2][2:1].f32
[N--D---] MMA.323216 R160[4:2][2:1].f32, R232[2:1][1:1].bf16.row, R221[1:1][2:1].bf16.row, R160[4:2][2:1].f32
[Ns-D-w-] MMA.323216 R168[4:2][2:1].f32, R232[2:1][1:1].bf16.row, R223[1:1][2:1].bf16.row, R168[4:2][2:1].f32
[N--D---] MMA.323216 R112[4:2][2:1].f32, R226[2:1][1:1].bf16.row, R209[1:1][2:1].bf16.row, R112[4:2][2:1].f32
[N--D---] MMA.323216 R120[4:2][2:1].f32, R226[2:1][1:1].bf16.row, R211[1:1][2:1].bf16.row, R120[4:2][2:1].f32
[Nw-D-s-]       LDMA.RD.B32.BST4  R[230:233], R7, X4, 24704, BURST_STRIDE=512
[N--D-s-]       LDMA.RD.B32.BST4  R[217:220], R3, X4, 36992, BURST_STRIDE=512
[N--D-s-]       LDMA.RD.B32.BST4  R[221:224], R3, X4, 39040, BURST_STRIDE=512
[N--D---] MMA.323216 R128[4:2][2:1].f32, R226[2:1][1:1].bf16.row, R213[1:1][2:1].bf16.row, R128[4:2][2:1].f32
[N--D---] MMA.323216 R136[4:2][2:1].f32, R226[2:1][1:1].bf16.row, R215[1:1][2:1].bf16.row, R136[4:2][2:1].f32
[N--D---] MMA.323216 R144[4:2][2:1].f32, R228[2:1][1:1].bf16.row, R209[1:1][2:1].bf16.row, R144[4:2][2:1].f32
[N--D---] MMA.323216 R152[4:2][2:1].f32, R228[2:1][1:1].bf16.row, R211[1:1][2:1].bf16.row, R152[4:2][2:1].f32
[N--D---] MMA.323216 R160[4:2][2:1].f32, R228[2:1][1:1].bf16.row, R213[1:1][2:1].bf16.row, R160[4:2][2:1].f32
[Ns-D-w-] MMA.323216 R168[4:2][2:1].f32, R228[2:1][1:1].bf16.row, R215[1:1][2:1].bf16.row, R168[4:2][2:1].f32
[N--D---] MMA.323216 R112[4:2][2:1].f32, R230[2:1][1:1].bf16.row, R217[1:1][2:1].bf16.row, R112[4:2][2:1].f32
[N--D---] MMA.323216 R120[4:2][2:1].f32, R230[2:1][1:1].bf16.row, R219[1:1][2:1].bf16.row, R120[4:2][2:1].f32
[N--D---] MMA.323216 R128[4:2][2:1].f32, R230[2:1][1:1].bf16.row, R221[1:1][2:1].bf16.row, R128[4:2][2:1].f32
[N--D---] MMA.323216 R136[4:2][2:1].f32, R230[2:1][1:1].bf16.row, R223[1:1][2:1].bf16.row, R136[4:2][2:1].f32
[N--D---] MMA.323216 R144[4:2][2:1].f32, R232[2:1][1:1].bf16.row, R217[1:1][2:1].bf16.row, R144[4:2][2:1].f32
[N--D---] MMA.323216 R152[4:2][2:1].f32, R232[2:1][1:1].bf16.row, R219[1:1][2:1].bf16.row, R152[4:2][2:1].f32
[N--D---] MMA.323216 R160[4:2][2:1].f32, R232[2:1][1:1].bf16.row, R221[1:1][2:1].bf16.row, R160[4:2][2:1].f32
[Ns-D-w-] MMA.323216 R168[4:2][2:1].f32, R232[2:1][1:1].bf16.row, R223[1:1][2:1].bf16.row, R168[4:2][2:1].f32
[NwwD---]      FOP.MUL R112, R112, R30
[N--D---]      FOP.MUL R113, R113, R30
[N--D---]      FOP.MUL R120, R120, R30
[N--D---]      FOP.MUL R121, R121, R30
[N--D---]      FOP.MUL R128, R128, R30
[N--D---]      FOP.MUL R129, R129, R30
[N--D---]      FOP.MUL R136, R136, R30
[N--D---]      FOP.MUL R137, R137, R30
[N--D---]      FOP.MUL R114, R114, R31
[N--D---]      FOP.MUL R115, R115, R31
[N--D---]      FOP.MUL R122, R122, R31
[N--D---]      FOP.MUL R123, R123, R31
[N--D---]      FOP.MUL R130, R130, R31
[N--D---]      FOP.MUL R131, R131, R31
[N--D---]      FOP.MUL R138, R138, R31
[N--D---]      FOP.MUL R139, R139, R31
[N--D---]      FOP.MUL R116, R116, R32
[N--D---]      FOP.MUL R117, R117, R32
[N--D---]      FOP.MUL R124, R124, R32
[N--D---]      FOP.MUL R125, R125, R32
[N--D---]      FOP.MUL R132, R132, R32
[N--D---]      FOP.MUL R133, R133, R32
[N--D---]      FOP.MUL R140, R140, R32
[N--D---]      FOP.MUL R141, R141, R32
[N--D---]      FOP.MUL R118, R118, R33
[N--D---]      FOP.MUL R119, R119, R33
[N--D---]      FOP.MUL R126, R126, R33
[N--D---]      FOP.MUL R127, R127, R33
[N--D---]      FOP.MUL R134, R134, R33
[N--D---]      FOP.MUL R135, R135, R33
[N--D---]      FOP.MUL R142, R142, R33
[N--D---]      FOP.MUL R143, R143, R33
[N--D---]      FOP.MUL R144, R144, R34
[N--D---]      FOP.MUL R145, R145, R34
[N--D---]      FOP.MUL R152, R152, R34
[N--D---]      FOP.MUL R153, R153, R34
[N--D---]      FOP.MUL R160, R160, R34
[N--D---]      FOP.MUL R161, R161, R34
[N--D---]      FOP.MUL R168, R168, R34
[N--D---]      FOP.MUL R169, R169, R34
[N--D---]      FOP.MUL R146, R146, R35
[N--D---]      FOP.MUL R147, R147, R35
[N--D---]      FOP.MUL R154, R154, R35
[N--D---]      FOP.MUL R155, R155, R35
[N--D---]      FOP.MUL R162, R162, R35
[N--D---]      FOP.MUL R163, R163, R35
[N--D---]      FOP.MUL R170, R170, R35
[N--D---]      FOP.MUL R171, R171, R35
[N--D---]      FOP.MUL R148, R148, R36
[N--D---]      FOP.MUL R149, R149, R36
[N--D---]      FOP.MUL R156, R156, R36
[N--D---]      FOP.MUL R157, R157, R36
[N--D---]      FOP.MUL R164, R164, R36
[N--D---]      FOP.MUL R165, R165, R36
[N--D---]      FOP.MUL R172, R172, R36
[N--D---]      FOP.MUL R173, R173, R36
[N--D---]      FOP.MUL R150, R150, R37
[N--D---]      FOP.MUL R151, R151, R37
[N--D---]      FOP.MUL R158, R158, R37
[N--D---]      FOP.MUL R159, R159, R37
[N--D---]      FOP.MUL R166, R166, R37
[N--D---]      FOP.MUL R167, R167, R37
[N--D---]      FOP.MUL R174, R174, R37
[N-sD---]      FOP.MUL R175, R175, R37
[NwwD-s-]    AF32.LOG2 R225, R225
[N--D--s]    MOV.LD.LM.B32 R49, LM5
[N--D--o]    MOV.LD.LM.B32 R48, LM6
[NwsD---]    BIT.ROT.LSR R37, C79, R0
[NswD---]    BIT.ROT.LSL R37, C45, R37
[NwsD---]    BIT.AND.LSR R42, C50, C78, R0
[NswD---]    INT.ADD R37.s32, R37.s32, R42.s32
[NwsD---]    INT.ADD R37.s32, R37.s32, SL2.s32
[NswD---]    INT.MADD R37.s32, SH77.s32, R37.s32, SL3.s32
[NwsD---]    BIT.AND.LSR R38, C80, SH76, R0
[NswD---]    INT.MUL R38.s32, C46.s32, R38.s32
[NwsD---]    BIT.LUT.AND R43, C59, R0
[NswD---]    INT.MUL R43.s32, C33.s32, R43.s32
[NwsD---]    INT.ADD R38.s32, R38.s32, R43.s32
[NswD---]    INT.ADD R44.s32, SH28.s32, R37.neg.s32
[NwsD---]    INT.ADD R45.s32, SH30.s32, R38.neg.s32
[NswD---]    INT.MADD R39.s32, SH49.s32, R38.s32, R37.s32
[NwsD---]    INT.MADD R39.s32, SH48.s32, R39.s32, R49.s32
[NswD---]    INT.MADD R39.s32, SH47.s32, R39.s32, R48.s32
[NwsD---]    BIT.ROT.LSL R39, C33, R39
[NswD-w-]    BIT.LUT.OR.TZ p0, SH4, SH5
[NwwD---]    CTRL.BR.ANYP0T OUTSL_END
[NwsD---]    BIT.ROT.LSR R46, C33, AR0
[NswD---]    FOP.ADD R225, R225, I0
[NwsD---]    FOP.MUL R225, R225, SH25
[NswD---]    INT.ADD R46.s32, SL2.s32, R46.s32
[NwsD---]    INT.BYP.TL p0, R46.s32, SH28.s32
[NswD---]    INT.MADD R47.s32, SH27.s32, R49.s32, R48.s32
[NwsD---]    INT.MADD R46.s32, SH28.s32, R46.s32, R47.s32
[NswD---]    BIT.ROT.LSL R46, C45, R46
[NwsD---]    BIT.LUT.AND.TZ.AP p0, C33, R0
[NswD---]  P0  DMA.ST.B32 R225, SH[4:5], R46
OUTSL_END:
[NwsD---]       INT.BYP.TLE p0, R44.s32, SH75.s32
[NswD---]       INT.BYP.TLE.OP p0, R45.s32, SH80.s32
[NwsD---]    INT.ADD SL0.s32, C0.s32, SH49.s32
[NwwD---]        CTRL.BR.ANYP0T MASK_STORE
[NswD---]    INT.MADD R40.s32,  C0.s32, R39.s32, SL0.s32
[NwsD---]    INT.ADD R40.s32, R40.s32, C0.s32
[NswD---]    FOP.MUL R112.l.bf16, C1, R112
[NwsD---]    DMA.ST.B16 R112.l, SH[0:1], R40
[NswD---]    INT.ADD R40.s32, R40.s32, SH72.s32
[NwsD---]    FOP.MUL R113.l.bf16, C1, R113
[NswD---]    DMA.ST.B16 R113.l, SH[0:1], R40
[NwsD---]    INT.MADD R40.s32,  C0.s32, R39.s32, SL0.s32
[NswD---]    INT.ADD R40.s32, R40.s32, SH77.s32
[NwsD---]    FOP.MUL R120.l.bf16, C1, R120
[NswD---]    DMA.ST.B16 R120.l, SH[0:1], R40
[NwsD---]    INT.ADD R40.s32, R40.s32, SH72.s32
[NswD---]    FOP.MUL R121.l.bf16, C1, R121
[NwsD---]    DMA.ST.B16 R121.l, SH[0:1], R40
[NswD---]    INT.MADD R40.s32,  C0.s32, R39.s32, SL0.s32
[NwsD---]    INT.ADD R40.s32, R40.s32, SH82.s32
[NswD---]    FOP.MUL R128.l.bf16, C1, R128
[NwsD---]    DMA.ST.B16 R128.l, SH[0:1], R40
[NswD---]    INT.ADD R40.s32, R40.s32, SH72.s32
[NwsD---]    FOP.MUL R129.l.bf16, C1, R129
[NswD---]    DMA.ST.B16 R129.l, SH[0:1], R40
[NwsD---]    INT.MADD R40.s32,  C0.s32, R39.s32, SL0.s32
[NswD---]    INT.ADD R40.s32, R40.s32, SH83.s32
[NwsD---]    FOP.MUL R136.l.bf16, C1, R136
[NswD---]    DMA.ST.B16 R136.l, SH[0:1], R40
[NwsD---]    INT.ADD R40.s32, R40.s32, SH72.s32
[NswD---]    FOP.MUL R137.l.bf16, C1, R137
[NwsD---]    DMA.ST.B16 R137.l, SH[0:1], R40
[NswD---]    INT.MADD R40.s32,  C39.s32, R39.s32, SL0.s32
[NwsD---]    INT.ADD R40.s32, R40.s32, C0.s32
[NswD---]    FOP.MUL R114.l.bf16, C1, R114
[NwsD---]    DMA.ST.B16 R114.l, SH[0:1], R40
[NswD---]    INT.ADD R40.s32, R40.s32, SH72.s32
[NwsD---]    FOP.MUL R115.l.bf16, C1, R115
[NswD---]    DMA.ST.B16 R115.l, SH[0:1], R40
[NwsD---]    INT.MADD R40.s32,  C39.s32, R39.s32, SL0.s32
[NswD---]    INT.ADD R40.s32, R40.s32, SH77.s32
[NwsD---]    FOP.MUL R122.l.bf16, C1, R122
[NswD---]    DMA.ST.B16 R122.l, SH[0:1], R40
[NwsD---]    INT.ADD R40.s32, R40.s32, SH72.s32
[NswD---]    FOP.MUL R123.l.bf16, C1, R123
[NwsD---]    DMA.ST.B16 R123.l, SH[0:1], R40
[NswD---]    INT.MADD R40.s32,  C39.s32, R39.s32, SL0.s32
[NwsD---]    INT.ADD R40.s32, R40.s32, SH82.s32
[NswD---]    FOP.MUL R130.l.bf16, C1, R130
[NwsD---]    DMA.ST.B16 R130.l, SH[0:1], R40
[NswD---]    INT.ADD R40.s32, R40.s32, SH72.s32
[NwsD---]    FOP.MUL R131.l.bf16, C1, R131
[NswD---]    DMA.ST.B16 R131.l, SH[0:1], R40
[NwsD---]    INT.MADD R40.s32,  C39.s32, R39.s32, SL0.s32
[NswD---]    INT.ADD R40.s32, R40.s32, SH83.s32
[NwsD---]    FOP.MUL R138.l.bf16, C1, R138
[NswD---]    DMA.ST.B16 R138.l, SH[0:1], R40
[NwsD---]    INT.ADD R40.s32, R40.s32, SH72.s32
[NswD---]    FOP.MUL R139.l.bf16, C1, R139
[NwsD---]    DMA.ST.B16 R139.l, SH[0:1], R40
[NswD---]    INT.MADD R40.s32,  SH72.s32, R39.s32, SL0.s32
[NwsD---]    INT.ADD R40.s32, R40.s32, C0.s32
[NswD---]    FOP.MUL R116.l.bf16, C1, R116
[NwsD---]    DMA.ST.B16 R116.l, SH[0:1], R40
[NswD---]    INT.ADD R40.s32, R40.s32, SH72.s32
[NwsD---]    FOP.MUL R117.l.bf16, C1, R117
[NswD---]    DMA.ST.B16 R117.l, SH[0:1], R40
[NwsD---]    INT.MADD R40.s32,  SH72.s32, R39.s32, SL0.s32
[NswD---]    INT.ADD R40.s32, R40.s32, SH77.s32
[NwsD---]    FOP.MUL R124.l.bf16, C1, R124
[NswD---]    DMA.ST.B16 R124.l, SH[0:1], R40
[NwsD---]    INT.ADD R40.s32, R40.s32, SH72.s32
[NswD---]    FOP.MUL R125.l.bf16, C1, R125
[NwsD---]    DMA.ST.B16 R125.l, SH[0:1], R40
[NswD---]    INT.MADD R40.s32,  SH72.s32, R39.s32, SL0.s32
[NwsD---]    INT.ADD R40.s32, R40.s32, SH82.s32
[NswD---]    FOP.MUL R132.l.bf16, C1, R132
[NwsD---]    DMA.ST.B16 R132.l, SH[0:1], R40
[NswD---]    INT.ADD R40.s32, R40.s32, SH72.s32
[NwsD---]    FOP.MUL R133.l.bf16, C1, R133
[NswD---]    DMA.ST.B16 R133.l, SH[0:1], R40
[NwsD---]    INT.MADD R40.s32,  SH72.s32, R39.s32, SL0.s32
[NswD---]    INT.ADD R40.s32, R40.s32, SH83.s32
[NwsD---]    FOP.MUL R140.l.bf16, C1, R140
[NswD---]    DMA.ST.B16 R140.l, SH[0:1], R40
[NwsD---]    INT.ADD R40.s32, R40.s32, SH72.s32
[NswD---]    FOP.MUL R141.l.bf16, C1, R141
[NwsD---]    DMA.ST.B16 R141.l, SH[0:1], R40
[NswD---]    INT.MADD R40.s32,  SH74.s32, R39.s32, SL0.s32
[NwsD---]    INT.ADD R40.s32, R40.s32, C0.s32
[NswD---]    FOP.MUL R118.l.bf16, C1, R118
[NwsD---]    DMA.ST.B16 R118.l, SH[0:1], R40
[NswD---]    INT.ADD R40.s32, R40.s32, SH72.s32
[NwsD---]    FOP.MUL R119.l.bf16, C1, R119
[NswD---]    DMA.ST.B16 R119.l, SH[0:1], R40
[NwsD---]    INT.MADD R40.s32,  SH74.s32, R39.s32, SL0.s32
[NswD---]    INT.ADD R40.s32, R40.s32, SH77.s32
[NwsD---]    FOP.MUL R126.l.bf16, C1, R126
[NswD---]    DMA.ST.B16 R126.l, SH[0:1], R40
[NwsD---]    INT.ADD R40.s32, R40.s32, SH72.s32
[NswD---]    FOP.MUL R127.l.bf16, C1, R127
[NwsD---]    DMA.ST.B16 R127.l, SH[0:1], R40
[NswD---]    INT.MADD R40.s32,  SH74.s32, R39.s32, SL0.s32
[NwsD---]    INT.ADD R40.s32, R40.s32, SH82.s32
[NswD---]    FOP.MUL R134.l.bf16, C1, R134
[NwsD---]    DMA.ST.B16 R134.l, SH[0:1], R40
[NswD---]    INT.ADD R40.s32, R40.s32, SH72.s32
[NwsD---]    FOP.MUL R135.l.bf16, C1, R135
[NswD---]    DMA.ST.B16 R135.l, SH[0:1], R40
[NwsD---]    INT.MADD R40.s32,  SH74.s32, R39.s32, SL0.s32
[NswD---]    INT.ADD R40.s32, R40.s32, SH83.s32
[NwsD---]    FOP.MUL R142.l.bf16, C1, R142
[NswD---]    DMA.ST.B16 R142.l, SH[0:1], R40
[NwsD---]    INT.ADD R40.s32, R40.s32, SH72.s32
[NswD---]    FOP.MUL R143.l.bf16, C1, R143
[NwsD---]    DMA.ST.B16 R143.l, SH[0:1], R40
[NswD---]    INT.MADD R40.s32,  SH77.s32, R39.s32, SL0.s32
[NwsD---]    INT.ADD R40.s32, R40.s32, C0.s32
[NswD---]    FOP.MUL R144.l.bf16, C1, R144
[NwsD---]    DMA.ST.B16 R144.l, SH[0:1], R40
[NswD---]    INT.ADD R40.s32, R40.s32, SH72.s32
[NwsD---]    FOP.MUL R145.l.bf16, C1, R145
[NswD---]    DMA.ST.B16 R145.l, SH[0:1], R40
[NwsD---]    INT.MADD R40.s32,  SH77.s32, R39.s32, SL0.s32
[NswD---]    INT.ADD R40.s32, R40.s32, SH77.s32
[NwsD---]    FOP.MUL R152.l.bf16, C1, R152
[NswD---]    DMA.ST.B16 R152.l, SH[0:1], R40
[NwsD---]    INT.ADD R40.s32, R40.s32, SH72.s32
[NswD---]    FOP.MUL R153.l.bf16, C1, R153
[NwsD---]    DMA.ST.B16 R153.l, SH[0:1], R40
[NswD---]    INT.MADD R40.s32,  SH77.s32, R39.s32, SL0.s32
[NwsD---]    INT.ADD R40.s32, R40.s32, SH82.s32
[NswD---]    FOP.MUL R160.l.bf16, C1, R160
[NwsD---]    DMA.ST.B16 R160.l, SH[0:1], R40
[NswD---]    INT.ADD R40.s32, R40.s32, SH72.s32
[NwsD---]    FOP.MUL R161.l.bf16, C1, R161
[NswD---]    DMA.ST.B16 R161.l, SH[0:1], R40
[NwsD---]    INT.MADD R40.s32,  SH77.s32, R39.s32, SL0.s32
[NswD---]    INT.ADD R40.s32, R40.s32, SH83.s32
[NwsD---]    FOP.MUL R168.l.bf16, C1, R168
[NswD---]    DMA.ST.B16 R168.l, SH[0:1], R40
[NwsD---]    INT.ADD R40.s32, R40.s32, SH72.s32
[NswD---]    FOP.MUL R169.l.bf16, C1, R169
[NwsD---]    DMA.ST.B16 R169.l, SH[0:1], R40
[NswD---]    INT.MADD R40.s32,  SH78.s32, R39.s32, SL0.s32
[NwsD---]    INT.ADD R40.s32, R40.s32, C0.s32
[NswD---]    FOP.MUL R146.l.bf16, C1, R146
[NwsD---]    DMA.ST.B16 R146.l, SH[0:1], R40
[NswD---]    INT.ADD R40.s32, R40.s32, SH72.s32
[NwsD---]    FOP.MUL R147.l.bf16, C1, R147
[NswD---]    DMA.ST.B16 R147.l, SH[0:1], R40
[NwsD---]    INT.MADD R40.s32,  SH78.s32, R39.s32, SL0.s32
[NswD---]    INT.ADD R40.s32, R40.s32, SH77.s32
[NwsD---]    FOP.MUL R154.l.bf16, C1, R154
[NswD---]    DMA.ST.B16 R154.l, SH[0:1], R40
[NwsD---]    INT.ADD R40.s32, R40.s32, SH72.s32
[NswD---]    FOP.MUL R155.l.bf16, C1, R155
[NwsD---]    DMA.ST.B16 R155.l, SH[0:1], R40
[NswD---]    INT.MADD R40.s32,  SH78.s32, R39.s32, SL0.s32
[NwsD---]    INT.ADD R40.s32, R40.s32, SH82.s32
[NswD---]    FOP.MUL R162.l.bf16, C1, R162
[NwsD---]    DMA.ST.B16 R162.l, SH[0:1], R40
[NswD---]    INT.ADD R40.s32, R40.s32, SH72.s32
[NwsD---]    FOP.MUL R163.l.bf16, C1, R163
[NswD---]    DMA.ST.B16 R163.l, SH[0:1], R40
[NwsD---]    INT.MADD R40.s32,  SH78.s32, R39.s32, SL0.s32
[NswD---]    INT.ADD R40.s32, R40.s32, SH83.s32
[NwsD---]    FOP.MUL R170.l.bf16, C1, R170
[NswD---]    DMA.ST.B16 R170.l, SH[0:1], R40
[NwsD---]    INT.ADD R40.s32, R40.s32, SH72.s32
[NswD---]    FOP.MUL R171.l.bf16, C1, R171
[NwsD---]    DMA.ST.B16 R171.l, SH[0:1], R40
[NswD---]    INT.MADD R40.s32,  SH79.s32, R39.s32, SL0.s32
[NwsD---]    INT.ADD R40.s32, R40.s32, C0.s32
[NswD---]    FOP.MUL R148.l.bf16, C1, R148
[NwsD---]    DMA.ST.B16 R148.l, SH[0:1], R40
[NswD---]    INT.ADD R40.s32, R40.s32, SH72.s32
[NwsD---]    FOP.MUL R149.l.bf16, C1, R149
[NswD---]    DMA.ST.B16 R149.l, SH[0:1], R40
[NwsD---]    INT.MADD R40.s32,  SH79.s32, R39.s32, SL0.s32
[NswD---]    INT.ADD R40.s32, R40.s32, SH77.s32
[NwsD---]    FOP.MUL R156.l.bf16, C1, R156
[NswD---]    DMA.ST.B16 R156.l, SH[0:1], R40
[NwsD---]    INT.ADD R40.s32, R40.s32, SH72.s32
[NswD---]    FOP.MUL R157.l.bf16, C1, R157
[NwsD---]    DMA.ST.B16 R157.l, SH[0:1], R40
[NswD---]    INT.MADD R40.s32,  SH79.s32, R39.s32, SL0.s32
[NwsD---]    INT.ADD R40.s32, R40.s32, SH82.s32
[NswD---]    FOP.MUL R164.l.bf16, C1, R164
[NwsD---]    DMA.ST.B16 R164.l, SH[0:1], R40
[NswD---]    INT.ADD R40.s32, R40.s32, SH72.s32
[NwsD---]    FOP.MUL R165.l.bf16, C1, R165
[NswD---]    DMA.ST.B16 R165.l, SH[0:1], R40
[NwsD---]    INT.MADD R40.s32,  SH79.s32, R39.s32, SL0.s32
[NswD---]    INT.ADD R40.s32, R40.s32, SH83.s32
[NwsD---]    FOP.MUL R172.l.bf16, C1, R172
[NswD---]    DMA.ST.B16 R172.l, SH[0:1], R40
[NwsD---]    INT.ADD R40.s32, R40.s32, SH72.s32
[NswD---]    FOP.MUL R173.l.bf16, C1, R173
[NwsD---]    DMA.ST.B16 R173.l, SH[0:1], R40
[NswD---]    INT.MADD R40.s32,  SH80.s32, R39.s32, SL0.s32
[NwsD---]    INT.ADD R40.s32, R40.s32, C0.s32
[NswD---]    FOP.MUL R150.l.bf16, C1, R150
[NwsD---]    DMA.ST.B16 R150.l, SH[0:1], R40
[NswD---]    INT.ADD R40.s32, R40.s32, SH72.s32
[NwsD---]    FOP.MUL R151.l.bf16, C1, R151
[NswD---]    DMA.ST.B16 R151.l, SH[0:1], R40
[NwsD---]    INT.MADD R40.s32,  SH80.s32, R39.s32, SL0.s32
[NswD---]    INT.ADD R40.s32, R40.s32, SH77.s32
[NwsD---]    FOP.MUL R158.l.bf16, C1, R158
[NswD---]    DMA.ST.B16 R158.l, SH[0:1], R40
[NwsD---]    INT.ADD R40.s32, R40.s32, SH72.s32
[NswD---]    FOP.MUL R159.l.bf16, C1, R159
[NwsD---]    DMA.ST.B16 R159.l, SH[0:1], R40
[NswD---]    INT.MADD R40.s32,  SH80.s32, R39.s32, SL0.s32
[NwsD---]    INT.ADD R40.s32, R40.s32, SH82.s32
[NswD---]    FOP.MUL R166.l.bf16, C1, R166
[NwsD---]    DMA.ST.B16 R166.l, SH[0:1], R40
[NswD---]    INT.ADD R40.s32, R40.s32, SH72.s32
[NwsD---]    FOP.MUL R167.l.bf16, C1, R167
[NswD---]    DMA.ST.B16 R167.l, SH[0:1], R40
[NwsD---]    INT.MADD R40.s32,  SH80.s32, R39.s32, SL0.s32
[NswD---]    INT.ADD R40.s32, R40.s32, SH83.s32
[NwsD---]    FOP.MUL R174.l.bf16, C1, R174
[NswD---]    DMA.ST.B16 R174.l, SH[0:1], R40
[NwsD---]    INT.ADD R40.s32, R40.s32, SH72.s32
[NswD---]    FOP.MUL R175.l.bf16, C1, R175
[NwsD---]    DMA.ST.B16 R175.l, SH[0:1], R40
[NwwDwww]    CTRL.NOP
[NwwD---]    AP.BARRIER LM0
[Nw-D---]    CTRL.END
MASK_STORE:
[NswD---]       INT.BYP.TG p0, R44.s32, C0.s32
[NwsD---]    INT.MADD R40.s32,  C0.s32, R39.s32, SL0.s32
[NswD---]    INT.ADD R40.s32, R40.s32, C0.s32
[NwsD---]    FOP.MUL R112.l.bf16, C1, R112
[NswD---]      INT.BYP.TG.AP p0, R45.s32, C0.s32
[NwsD---]  P0  DMA.ST.B16 R112.l, SH[0:1], R40
[NswD---]    INT.ADD R40.s32, R40.s32, SH72.s32
[NwsD---]    FOP.MUL R113.l.bf16, C1, R113
[NswD---]      INT.BYP.TG.AP p0, R45.s32, C39.s32
[NwsD---]  P0  DMA.ST.B16 R113.l, SH[0:1], R40
[NswD---]    INT.MADD R40.s32,  C0.s32, R39.s32, SL0.s32
[NwsD---]    INT.ADD R40.s32, R40.s32, SH77.s32
[NswD---]    FOP.MUL R120.l.bf16, C1, R120
[NwsD---]      INT.BYP.TG.AP p0, R45.s32, SH72.s32
[NswD---]  P0  DMA.ST.B16 R120.l, SH[0:1], R40
[NwsD---]    INT.ADD R40.s32, R40.s32, SH72.s32
[NswD---]    FOP.MUL R121.l.bf16, C1, R121
[NwsD---]      INT.BYP.TG.AP p0, R45.s32, SH74.s32
[NswD---]  P0  DMA.ST.B16 R121.l, SH[0:1], R40
[NwsD---]    INT.MADD R40.s32,  C0.s32, R39.s32, SL0.s32
[NswD---]    INT.ADD R40.s32, R40.s32, SH82.s32
[NwsD---]    FOP.MUL R128.l.bf16, C1, R128
[NswD---]      INT.BYP.TG.AP p0, R45.s32, SH77.s32
[NwsD---]  P0  DMA.ST.B16 R128.l, SH[0:1], R40
[NswD---]    INT.ADD R40.s32, R40.s32, SH72.s32
[NwsD---]    FOP.MUL R129.l.bf16, C1, R129
[NswD---]      INT.BYP.TG.AP p0, R45.s32, SH78.s32
[NwsD---]  P0  DMA.ST.B16 R129.l, SH[0:1], R40
[NswD---]    INT.MADD R40.s32,  C0.s32, R39.s32, SL0.s32
[NwsD---]    INT.ADD R40.s32, R40.s32, SH83.s32
[NswD---]    FOP.MUL R136.l.bf16, C1, R136
[NwsD---]      INT.BYP.TG.AP p0, R45.s32, SH79.s32
[NswD---]  P0  DMA.ST.B16 R136.l, SH[0:1], R40
[NwsD---]    INT.ADD R40.s32, R40.s32, SH72.s32
[NswD---]    FOP.MUL R137.l.bf16, C1, R137
[NwsD---]      INT.BYP.TG.AP p0, R45.s32, SH80.s32
[NswD---]  P0  DMA.ST.B16 R137.l, SH[0:1], R40
[NwsD---]       INT.BYP.TG p0, R44.s32, C46.s32
[NswD---]    INT.MADD R40.s32,  C39.s32, R39.s32, SL0.s32
[NwsD---]    INT.ADD R40.s32, R40.s32, C0.s32
[NswD---]    FOP.MUL R114.l.bf16, C1, R114
[NwsD---]      INT.BYP.TG.AP p0, R45.s32, C0.s32
[NswD---]  P0  DMA.ST.B16 R114.l, SH[0:1], R40
[NwsD---]    INT.ADD R40.s32, R40.s32, SH72.s32
[NswD---]    FOP.MUL R115.l.bf16, C1, R115
[NwsD---]      INT.BYP.TG.AP p0, R45.s32, C39.s32
[NswD---]  P0  DMA.ST.B16 R115.l, SH[0:1], R40
[NwsD---]    INT.MADD R40.s32,  C39.s32, R39.s32, SL0.s32
[NswD---]    INT.ADD R40.s32, R40.s32, SH77.s32
[NwsD---]    FOP.MUL R122.l.bf16, C1, R122
[NswD---]      INT.BYP.TG.AP p0, R45.s32, SH72.s32
[NwsD---]  P0  DMA.ST.B16 R122.l, SH[0:1], R40
[NswD---]    INT.ADD R40.s32, R40.s32, SH72.s32
[NwsD---]    FOP.MUL R123.l.bf16, C1, R123
[NswD---]      INT.BYP.TG.AP p0, R45.s32, SH74.s32
[NwsD---]  P0  DMA.ST.B16 R123.l, SH[0:1], R40
[NswD---]    INT.MADD R40.s32,  C39.s32, R39.s32, SL0.s32
[NwsD---]    INT.ADD R40.s32, R40.s32, SH82.s32
[NswD---]    FOP.MUL R130.l.bf16, C1, R130
[NwsD---]      INT.BYP.TG.AP p0, R45.s32, SH77.s32
[NswD---]  P0  DMA.ST.B16 R130.l, SH[0:1], R40
[NwsD---]    INT.ADD R40.s32, R40.s32, SH72.s32
[NswD---]    FOP.MUL R131.l.bf16, C1, R131
[NwsD---]      INT.BYP.TG.AP p0, R45.s32, SH78.s32
[NswD---]  P0  DMA.ST.B16 R131.l, SH[0:1], R40
[NwsD---]    INT.MADD R40.s32,  C39.s32, R39.s32, SL0.s32
[NswD---]    INT.ADD R40.s32, R40.s32, SH83.s32
[NwsD---]    FOP.MUL R138.l.bf16, C1, R138
[NswD---]      INT.BYP.TG.AP p0, R45.s32, SH79.s32
[NwsD---]  P0  DMA.ST.B16 R138.l, SH[0:1], R40
[NswD---]    INT.ADD R40.s32, R40.s32, SH72.s32
[NwsD---]    FOP.MUL R139.l.bf16, C1, R139
[NswD---]      INT.BYP.TG.AP p0, R45.s32, SH80.s32
[NwsD---]  P0  DMA.ST.B16 R139.l, SH[0:1], R40
[NswD---]       INT.BYP.TG p0, R44.s32, C39.s32
[NwsD---]    INT.MADD R40.s32,  SH72.s32, R39.s32, SL0.s32
[NswD---]    INT.ADD R40.s32, R40.s32, C0.s32
[NwsD---]    FOP.MUL R116.l.bf16, C1, R116
[NswD---]      INT.BYP.TG.AP p0, R45.s32, C0.s32
[NwsD---]  P0  DMA.ST.B16 R116.l, SH[0:1], R40
[NswD---]    INT.ADD R40.s32, R40.s32, SH72.s32
[NwsD---]    FOP.MUL R117.l.bf16, C1, R117
[NswD---]      INT.BYP.TG.AP p0, R45.s32, C39.s32
[NwsD---]  P0  DMA.ST.B16 R117.l, SH[0:1], R40
[NswD---]    INT.MADD R40.s32,  SH72.s32, R39.s32, SL0.s32
[NwsD---]    INT.ADD R40.s32, R40.s32, SH77.s32
[NswD---]    FOP.MUL R124.l.bf16, C1, R124
[NwsD---]      INT.BYP.TG.AP p0, R45.s32, SH72.s32
[NswD---]  P0  DMA.ST.B16 R124.l, SH[0:1], R40
[NwsD---]    INT.ADD R40.s32, R40.s32, SH72.s32
[NswD---]    FOP.MUL R125.l.bf16, C1, R125
[NwsD---]      INT.BYP.TG.AP p0, R45.s32, SH74.s32
[NswD---]  P0  DMA.ST.B16 R125.l, SH[0:1], R40
[NwsD---]    INT.MADD R40.s32,  SH72.s32, R39.s32, SL0.s32
[NswD---]    INT.ADD R40.s32, R40.s32, SH82.s32
[NwsD---]    FOP.MUL R132.l.bf16, C1, R132
[NswD---]      INT.BYP.TG.AP p0, R45.s32, SH77.s32
[NwsD---]  P0  DMA.ST.B16 R132.l, SH[0:1], R40
[NswD---]    INT.ADD R40.s32, R40.s32, SH72.s32
[NwsD---]    FOP.MUL R133.l.bf16, C1, R133
[NswD---]      INT.BYP.TG.AP p0, R45.s32, SH78.s32
[NwsD---]  P0  DMA.ST.B16 R133.l, SH[0:1], R40
[NswD---]    INT.MADD R40.s32,  SH72.s32, R39.s32, SL0.s32
[NwsD---]    INT.ADD R40.s32, R40.s32, SH83.s32
[NswD---]    FOP.MUL R140.l.bf16, C1, R140
[NwsD---]      INT.BYP.TG.AP p0, R45.s32, SH79.s32
[NswD---]  P0  DMA.ST.B16 R140.l, SH[0:1], R40
[NwsD---]    INT.ADD R40.s32, R40.s32, SH72.s32
[NswD---]    FOP.MUL R141.l.bf16, C1, R141
[NwsD---]      INT.BYP.TG.AP p0, R45.s32, SH80.s32
[NswD---]  P0  DMA.ST.B16 R141.l, SH[0:1], R40
[NwsD---]       INT.BYP.TG p0, R44.s32, C48.s32
[NswD---]    INT.MADD R40.s32,  SH74.s32, R39.s32, SL0.s32
[NwsD---]    INT.ADD R40.s32, R40.s32, C0.s32
[NswD---]    FOP.MUL R118.l.bf16, C1, R118
[NwsD---]      INT.BYP.TG.AP p0, R45.s32, C0.s32
[NswD---]  P0  DMA.ST.B16 R118.l, SH[0:1], R40
[NwsD---]    INT.ADD R40.s32, R40.s32, SH72.s32
[NswD---]    FOP.MUL R119.l.bf16, C1, R119
[NwsD---]      INT.BYP.TG.AP p0, R45.s32, C39.s32
[NswD---]  P0  DMA.ST.B16 R119.l, SH[0:1], R40
[NwsD---]    INT.MADD R40.s32,  SH74.s32, R39.s32, SL0.s32
[NswD---]    INT.ADD R40.s32, R40.s32, SH77.s32
[NwsD---]    FOP.MUL R126.l.bf16, C1, R126
[NswD---]      INT.BYP.TG.AP p0, R45.s32, SH72.s32
[NwsD---]  P0  DMA.ST.B16 R126.l, SH[0:1], R40
[NswD---]    INT.ADD R40.s32, R40.s32, SH72.s32
[NwsD---]    FOP.MUL R127.l.bf16, C1, R127
[NswD---]      INT.BYP.TG.AP p0, R45.s32, SH74.s32
[NwsD---]  P0  DMA.ST.B16 R127.l, SH[0:1], R40
[NswD---]    INT.MADD R40.s32,  SH74.s32, R39.s32, SL0.s32
[NwsD---]    INT.ADD R40.s32, R40.s32, SH82.s32
[NswD---]    FOP.MUL R134.l.bf16, C1, R134
[NwsD---]      INT.BYP.TG.AP p0, R45.s32, SH77.s32
[NswD---]  P0  DMA.ST.B16 R134.l, SH[0:1], R40
[NwsD---]    INT.ADD R40.s32, R40.s32, SH72.s32
[NswD---]    FOP.MUL R135.l.bf16, C1, R135
[NwsD---]      INT.BYP.TG.AP p0, R45.s32, SH78.s32
[NswD---]  P0  DMA.ST.B16 R135.l, SH[0:1], R40
[NwsD---]    INT.MADD R40.s32,  SH74.s32, R39.s32, SL0.s32
[NswD---]    INT.ADD R40.s32, R40.s32, SH83.s32
[NwsD---]    FOP.MUL R142.l.bf16, C1, R142
[NswD---]      INT.BYP.TG.AP p0, R45.s32, SH79.s32
[NwsD---]  P0  DMA.ST.B16 R142.l, SH[0:1], R40
[NswD---]    INT.ADD R40.s32, R40.s32, SH72.s32
[NwsD---]    FOP.MUL R143.l.bf16, C1, R143
[NswD---]      INT.BYP.TG.AP p0, R45.s32, SH80.s32
[NwsD---]  P0  DMA.ST.B16 R143.l, SH[0:1], R40
[NswD---]       INT.BYP.TG p0, R44.s32, SH72.s32
[NwsD---]    INT.MADD R40.s32,  SH77.s32, R39.s32, SL0.s32
[NswD---]    INT.ADD R40.s32, R40.s32, C0.s32
[NwsD---]    FOP.MUL R144.l.bf16, C1, R144
[NswD---]      INT.BYP.TG.AP p0, R45.s32, C0.s32
[NwsD---]  P0  DMA.ST.B16 R144.l, SH[0:1], R40
[NswD---]    INT.ADD R40.s32, R40.s32, SH72.s32
[NwsD---]    FOP.MUL R145.l.bf16, C1, R145
[NswD---]      INT.BYP.TG.AP p0, R45.s32, C39.s32
[NwsD---]  P0  DMA.ST.B16 R145.l, SH[0:1], R40
[NswD---]    INT.MADD R40.s32,  SH77.s32, R39.s32, SL0.s32
[NwsD---]    INT.ADD R40.s32, R40.s32, SH77.s32
[NswD---]    FOP.MUL R152.l.bf16, C1, R152
[NwsD---]      INT.BYP.TG.AP p0, R45.s32, SH72.s32
[NswD---]  P0  DMA.ST.B16 R152.l, SH[0:1], R40
[NwsD---]    INT.ADD R40.s32, R40.s32, SH72.s32
[NswD---]    FOP.MUL R153.l.bf16, C1, R153
[NwsD---]      INT.BYP.TG.AP p0, R45.s32, SH74.s32
[NswD---]  P0  DMA.ST.B16 R153.l, SH[0:1], R40
[NwsD---]    INT.MADD R40.s32,  SH77.s32, R39.s32, SL0.s32
[NswD---]    INT.ADD R40.s32, R40.s32, SH82.s32
[NwsD---]    FOP.MUL R160.l.bf16, C1, R160
[NswD---]      INT.BYP.TG.AP p0, R45.s32, SH77.s32
[NwsD---]  P0  DMA.ST.B16 R160.l, SH[0:1], R40
[NswD---]    INT.ADD R40.s32, R40.s32, SH72.s32
[NwsD---]    FOP.MUL R161.l.bf16, C1, R161
[NswD---]      INT.BYP.TG.AP p0, R45.s32, SH78.s32
[NwsD---]  P0  DMA.ST.B16 R161.l, SH[0:1], R40
[NswD---]    INT.MADD R40.s32,  SH77.s32, R39.s32, SL0.s32
[NwsD---]    INT.ADD R40.s32, R40.s32, SH83.s32
[NswD---]    FOP.MUL R168.l.bf16, C1, R168
[NwsD---]      INT.BYP.TG.AP p0, R45.s32, SH79.s32
[NswD---]  P0  DMA.ST.B16 R168.l, SH[0:1], R40
[NwsD---]    INT.ADD R40.s32, R40.s32, SH72.s32
[NswD---]    FOP.MUL R169.l.bf16, C1, R169
[NwsD---]      INT.BYP.TG.AP p0, R45.s32, SH80.s32
[NswD---]  P0  DMA.ST.B16 R169.l, SH[0:1], R40
[NwsD---]       INT.BYP.TG p0, R44.s32, SH73.s32
[NswD---]    INT.MADD R40.s32,  SH78.s32, R39.s32, SL0.s32
[NwsD---]    INT.ADD R40.s32, R40.s32, C0.s32
[NswD---]    FOP.MUL R146.l.bf16, C1, R146
[NwsD---]      INT.BYP.TG.AP p0, R45.s32, C0.s32
[NswD---]  P0  DMA.ST.B16 R146.l, SH[0:1], R40
[NwsD---]    INT.ADD R40.s32, R40.s32, SH72.s32
[NswD---]    FOP.MUL R147.l.bf16, C1, R147
[NwsD---]      INT.BYP.TG.AP p0, R45.s32, C39.s32
[NswD---]  P0  DMA.ST.B16 R147.l, SH[0:1], R40
[NwsD---]    INT.MADD R40.s32,  SH78.s32, R39.s32, SL0.s32
[NswD---]    INT.ADD R40.s32, R40.s32, SH77.s32
[NwsD---]    FOP.MUL R154.l.bf16, C1, R154
[NswD---]      INT.BYP.TG.AP p0, R45.s32, SH72.s32
[NwsD---]  P0  DMA.ST.B16 R154.l, SH[0:1], R40
[NswD---]    INT.ADD R40.s32, R40.s32, SH72.s32
[NwsD---]    FOP.MUL R155.l.bf16, C1, R155
[NswD---]      INT.BYP.TG.AP p0, R45.s32, SH74.s32
[NwsD---]  P0  DMA.ST.B16 R155.l, SH[0:1], R40
[NswD---]    INT.MADD R40.s32,  SH78.s32, R39.s32, SL0.s32
[NwsD---]    INT.ADD R40.s32, R40.s32, SH82.s32
[NswD---]    FOP.MUL R162.l.bf16, C1, R162
[NwsD---]      INT.BYP.TG.AP p0, R45.s32, SH77.s32
[NswD---]  P0  DMA.ST.B16 R162.l, SH[0:1], R40
[NwsD---]    INT.ADD R40.s32, R40.s32, SH72.s32
[NswD---]    FOP.MUL R163.l.bf16, C1, R163
[NwsD---]      INT.BYP.TG.AP p0, R45.s32, SH78.s32
[NswD---]  P0  DMA.ST.B16 R163.l, SH[0:1], R40
[NwsD---]    INT.MADD R40.s32,  SH78.s32, R39.s32, SL0.s32
[NswD---]    INT.ADD R40.s32, R40.s32, SH83.s32
[NwsD---]    FOP.MUL R170.l.bf16, C1, R170
[NswD---]      INT.BYP.TG.AP p0, R45.s32, SH79.s32
[NwsD---]  P0  DMA.ST.B16 R170.l, SH[0:1], R40
[NswD---]    INT.ADD R40.s32, R40.s32, SH72.s32
[NwsD---]    FOP.MUL R171.l.bf16, C1, R171
[NswD---]      INT.BYP.TG.AP p0, R45.s32, SH80.s32
[NwsD---]  P0  DMA.ST.B16 R171.l, SH[0:1], R40
[NswD---]       INT.BYP.TG p0, R44.s32, SH74.s32
[NwsD---]    INT.MADD R40.s32,  SH79.s32, R39.s32, SL0.s32
[NswD---]    INT.ADD R40.s32, R40.s32, C0.s32
[NwsD---]    FOP.MUL R148.l.bf16, C1, R148
[NswD---]      INT.BYP.TG.AP p0, R45.s32, C0.s32
[NwsD---]  P0  DMA.ST.B16 R148.l, SH[0:1], R40
[NswD---]    INT.ADD R40.s32, R40.s32, SH72.s32
[NwsD---]    FOP.MUL R149.l.bf16, C1, R149
[NswD---]      INT.BYP.TG.AP p0, R45.s32, C39.s32
[NwsD---]  P0  DMA.ST.B16 R149.l, SH[0:1], R40
[NswD---]    INT.MADD R40.s32,  SH79.s32, R39.s32, SL0.s32
[NwsD---]    INT.ADD R40.s32, R40.s32, SH77.s32
[NswD---]    FOP.MUL R156.l.bf16, C1, R156
[NwsD---]      INT.BYP.TG.AP p0, R45.s32, SH72.s32
[NswD---]  P0  DMA.ST.B16 R156.l, SH[0:1], R40
[NwsD---]    INT.ADD R40.s32, R40.s32, SH72.s32
[NswD---]    FOP.MUL R157.l.bf16, C1, R157
[NwsD---]      INT.BYP.TG.AP p0, R45.s32, SH74.s32
[NswD---]  P0  DMA.ST.B16 R157.l, SH[0:1], R40
[NwsD---]    INT.MADD R40.s32,  SH79.s32, R39.s32, SL0.s32
[NswD---]    INT.ADD R40.s32, R40.s32, SH82.s32
[NwsD---]    FOP.MUL R164.l.bf16, C1, R164
[NswD---]      INT.BYP.TG.AP p0, R45.s32, SH77.s32
[NwsD---]  P0  DMA.ST.B16 R164.l, SH[0:1], R40
[NswD---]    INT.ADD R40.s32, R40.s32, SH72.s32
[NwsD---]    FOP.MUL R165.l.bf16, C1, R165
[NswD---]      INT.BYP.TG.AP p0, R45.s32, SH78.s32
[NwsD---]  P0  DMA.ST.B16 R165.l, SH[0:1], R40
[NswD---]    INT.MADD R40.s32,  SH79.s32, R39.s32, SL0.s32
[NwsD---]    INT.ADD R40.s32, R40.s32, SH83.s32
[NswD---]    FOP.MUL R172.l.bf16, C1, R172
[NwsD---]      INT.BYP.TG.AP p0, R45.s32, SH79.s32
[NswD---]  P0  DMA.ST.B16 R172.l, SH[0:1], R40
[NwsD---]    INT.ADD R40.s32, R40.s32, SH72.s32
[NswD---]    FOP.MUL R173.l.bf16, C1, R173
[NwsD---]      INT.BYP.TG.AP p0, R45.s32, SH80.s32
[NswD---]  P0  DMA.ST.B16 R173.l, SH[0:1], R40
[NwsD---]       INT.BYP.TG p0, R44.s32, SH75.s32
[NswD---]    INT.MADD R40.s32,  SH80.s32, R39.s32, SL0.s32
[NwsD---]    INT.ADD R40.s32, R40.s32, C0.s32
[NswD---]    FOP.MUL R150.l.bf16, C1, R150
[NwsD---]      INT.BYP.TG.AP p0, R45.s32, C0.s32
[NswD---]  P0  DMA.ST.B16 R150.l, SH[0:1], R40
[NwsD---]    INT.ADD R40.s32, R40.s32, SH72.s32
[NswD---]    FOP.MUL R151.l.bf16, C1, R151
[NwsD---]      INT.BYP.TG.AP p0, R45.s32, C39.s32
[NswD---]  P0  DMA.ST.B16 R151.l, SH[0:1], R40
[NwsD---]    INT.MADD R40.s32,  SH80.s32, R39.s32, SL0.s32
[NswD---]    INT.ADD R40.s32, R40.s32, SH77.s32
[NwsD---]    FOP.MUL R158.l.bf16, C1, R158
[NswD---]      INT.BYP.TG.AP p0, R45.s32, SH72.s32
[NwsD---]  P0  DMA.ST.B16 R158.l, SH[0:1], R40
[NswD---]    INT.ADD R40.s32, R40.s32, SH72.s32
[NwsD---]    FOP.MUL R159.l.bf16, C1, R159
[NswD---]      INT.BYP.TG.AP p0, R45.s32, SH74.s32
[NwsD---]  P0  DMA.ST.B16 R159.l, SH[0:1], R40
[NswD---]    INT.MADD R40.s32,  SH80.s32, R39.s32, SL0.s32
[NwsD---]    INT.ADD R40.s32, R40.s32, SH82.s32
[NswD---]    FOP.MUL R166.l.bf16, C1, R166
[NwsD---]      INT.BYP.TG.AP p0, R45.s32, SH77.s32
[NswD---]  P0  DMA.ST.B16 R166.l, SH[0:1], R40
[NwsD---]    INT.ADD R40.s32, R40.s32, SH72.s32
[NswD---]    FOP.MUL R167.l.bf16, C1, R167
[NwsD---]      INT.BYP.TG.AP p0, R45.s32, SH78.s32
[NswD---]  P0  DMA.ST.B16 R167.l, SH[0:1], R40
[NwsD---]    INT.MADD R40.s32,  SH80.s32, R39.s32, SL0.s32
[NswD---]    INT.ADD R40.s32, R40.s32, SH83.s32
[NwsD---]    FOP.MUL R174.l.bf16, C1, R174
[NswD---]      INT.BYP.TG.AP p0, R45.s32, SH79.s32
[NwsD---]  P0  DMA.ST.B16 R174.l, SH[0:1], R40
[NswD---]    INT.ADD R40.s32, R40.s32, SH72.s32
[NwsD---]    FOP.MUL R175.l.bf16, C1, R175
[NswD---]      INT.BYP.TG.AP p0, R45.s32, SH80.s32
[NwsD---]  P0  DMA.ST.B16 R175.l, SH[0:1], R40
[NwwDwww]    CTRL.NOP
[NwwD---]    AP.BARRIER LM0
[Nw-D---]    CTRL.END

.Lfunc_end0:
	.size	apxinf_qy2_bf16flash_atten_512_256x64x128, .Lfunc_end0-apxinf_qy2_bf16flash_atten_512_256x64x128




    .type	llvm.mtgpu.kernel.apxinf_qy2_bf16flash_atten_512_256x64x128.sm,@object
    .local	llvm.mtgpu.kernel.apxinf_qy2_bf16flash_atten_512_256x64x128.sm
    .comm	llvm.mtgpu.kernel.apxinf_qy2_bf16flash_atten_512_256x64x128.sm,57344,4

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
          .name:           output_ptr
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
          .name:           outsl_ptr
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
          .name:           input_q
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
          .name:           input_k
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
          .name:           input_v
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
          .name:           input_mask
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
          .name:           scale
          .offset:         24
          .reg_bank:       3
          .size:           4
          .type:           8
        .input_mode:     0
        .kind:           18
      - .argBase:
          .address_space:  0
          .align:          4
          .isVolatile:     false
          .layout:         0
          .name:           nr_ln2
          .offset:         25
          .reg_bank:       3
          .size:           4
          .type:           8
        .input_mode:     0
        .kind:           18
      - .argBase:
          .address_space:  0
          .align:          4
          .isVolatile:     false
          .layout:         0
          .name:           batch_size
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
          .name:           nheads
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
          .name:           seq_q_len
          .offset:         28
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
          .name:           seq_kv_len
          .offset:         29
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
          .name:           head_dim
          .offset:         30
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
          .name:           head_dim_sub1
          .offset:         31
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
          .name:           batch_stride_q
          .offset:         32
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
          .name:           head_stride_q
          .offset:         33
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
          .name:           seq_stride_q
          .offset:         34
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
          .name:           batch_stride_k
          .offset:         35
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
          .name:           head_stride_k
          .offset:         36
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
          .name:           seq_stride_k
          .offset:         37
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
          .name:           batch_stride_v
          .offset:         38
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
          .name:           head_stride_v
          .offset:         39
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
          .name:           seq_stride_v
          .offset:         40
          .reg_bank:       3
          .size:           4
          .type:           6
        .input_mode:     0
        .kind:           18
      - .argBase:
          .address_space:  0
          .align:          8
          .isVolatile:     false
          .layout:         0
          .name:           batch_stride_mask
          .offset:         42
          .reg_bank:       3
          .size:           8
          .type:           9
        .input_mode:     0
        .kind:           18
      - .argBase:
          .address_space:  0
          .align:          8
          .isVolatile:     false
          .layout:         0
          .name:           head_stride_mask
          .offset:         44
          .reg_bank:       3
          .size:           8
          .type:           9
        .input_mode:     0
        .kind:           18
      - .argBase:
          .address_space:  0
          .align:          4
          .isVolatile:     false
          .layout:         0
          .name:           seq_stride_mask
          .offset:         46
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
          .name:           batch_stride_out
          .offset:         47
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
          .name:           head_stride_out
          .offset:         48
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
          .name:           seq_stride_out
          .offset:         49
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
          .name:           rln2_scale
          .offset:         50
          .reg_bank:       3
          .size:           4
          .type:           8
        .input_mode:     0
        .kind:           18
      - .argBase:
          .address_space:  0
          .align:          4
          .isVolatile:     false
          .layout:         0
          .name:           kvheads
          .offset:         51
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
          .name:           head_offset
          .offset:         52
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
          .name:           head_group_ori
          .offset:         53
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
          .name:           head_group_mul
          .offset:         54
          .reg_bank:       3
          .size:           4
          .type:           7
        .input_mode:     0
        .kind:           18
      - .argBase:
          .address_space:  0
          .align:          4
          .isVolatile:     false
          .layout:         0
          .name:           head_group_sft
          .offset:         55
          .reg_bank:       3
          .size:           4
          .type:           7
        .input_mode:     0
        .kind:           18
      - .argBase:
          .address_space:  0
          .align:          4
          .isVolatile:     false
          .layout:         0
          .name:           ldg_k_rewind
          .offset:         56
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
          .name:           ldg_v_stride
          .offset:         57
          .reg_bank:       3
          .size:           4
          .type:           6
        .input_mode:     0
        .kind:           18
      - .argBase:
          .address_space:  0
          .align:          0
          .isVolatile:     false
          .layout:         0
          .name:           prog_addr
          .offset:         58
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
          .name:           prog_size
          .offset:         60
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
          .name:           gridDim_x
          .offset:         61
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
          .offset:         62
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
          .offset:         63
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
          .offset:         64
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
          .offset:         65
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
          .offset:         67
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
    .attr_reg_count: 6
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
      .offset:         68
      .reg_bank:       3
      .size:           72
      .values:
        - 10
        - 11
        - 12
        - 13
        - 32
        - 40
        - 48
        - 56
        - 63
        - 64
        - 80
        - 96
        - 112
        - 127
        - 128
        - 192
        - 256
        - 3379831792
    .indirect_call_flag: false
    .call_flag: false
    .internal_reg_count: 4
    .local_mem_used_in_cluster: 0
    .max_block_size: 1
    .name:           apxinf_qy2_bf16flash_atten_512_256x64x128
    .private_memory_size: 0
    .shared_memory_size: 57472
    .shared_reg_count: 86
    .slot_reg_count: 4
    .temp_reg_count: 244
    .wave_mode:      0
mtgpu.version:
  - 2
  - 2
...

	.mtgpu_metadata.end

    