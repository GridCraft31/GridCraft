# -*- coding: utf-8 -*-
"""GFL กับ GFM เมื่อความแข็งของกริดเปลี่ยนไป จำลองด้วย DPsim (EMT สามเฟส)

ระบบ: แหล่งจ่ายกริด 22 kV -> อิมพีแดนซ์กริด (ตาม SCR) -> บัส 22 kV (มีโหลด 500 kW รอต่อ)
      -> หม้อแปลง step up 1 MVA 6% -> อินเวอร์เตอร์ 1 MVA จ่าย 800 kW
ทุกอย่างอ้างไปฝั่ง 380 V ในการจำลอง (หม้อแปลงแทนด้วยอิมพีแดนซ์ 6%)

GFM ใช้โมเดล SSN_GFM ของ DPsim
GFL ใช้โครงข่ายและตัวกรองใน DPsim แต่วงควบคุมเขียนเองใน Python สั่งแหล่งแรงดันทุก time step
เพราะโมเดล GFL ที่มากับ DPsim 1.4.0 คุมกระแสได้ราว 13 Hz และไม่มี feedforward แรงดัน

บน Colab ให้ติดตั้งก่อน:  !pip install dpsim
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
import dpsimpy as dp

# =====================================================================
# ตัวแปรที่ลองปรับได้
# =====================================================================
QUICK = False           # False รันครบทุกจุดแล้วได้รูปเหมือนในโพสต์ (ราว 10 นาที) · True รันแถวตัวอย่างพอให้เห็นภาพ (ราว 1 นาที)
SCR_LIST = [1000, 300, 100, 50, 20, 10, 5, 3, 2.5, 2, 1.8]
PLL_LIST = [1, 2, 4, 6, 8, 10, 12]    # ความเร็ว PLL ของ GFL เป็นเท่าของค่าตั้งต้น
PLL_LIST_Z = [1, 4, 8, 12]            # เส้น damping ratio ที่วาดในรูป 4
QUICK_PLL, QUICK_XV = [10], [0.0]     # แถวที่รันเมื่อ QUICK = True
XV_LIST = [0.0, 0.01, 0.026]          # virtual reactance ของ GFM (pu)
JUMP_DEG = -5.0                       # มุมเฟสกริดกระโดด (องศา)

F = 50.0; W = 2 * np.pi * F
DT, T_END, T_STEP = 50e-6, 3.0, 1.0   # time step, เวลาจำลอง, เวลาที่ต่อโหลด
S = 1e6                               # พิกัดอินเวอร์เตอร์ 1 MVA
VLV = np.sqrt(3) * 220.0              # 380 V
VPK = np.sqrt(2) * 220.0
XR, ZT, XRT = 10.0, 0.06, 8.0         # X/R กริด, อิมพีแดนซ์หม้อแปลง (pu), X/R หม้อแปลง
P_SET = 0.8 * S
LOAD = 0.5 * S                        # โหลดที่ต่อที่ T_STEP (ตรวจแล้วขนาดโหลดไม่เปลี่ยนสถานะของจุดในแผนที่)

# ฮาร์ดแวร์ชุดเดียวกันทั้งสองแบบ (ค่าจากตัวอย่าง DPsim 15 kVA สเกลเป็น 1 MVA ให้ pu เท่าเดิม)
K = S / 15e3
LF, CF, RF, RC = 3e-3 / K, 20e-6 * K, 0.05 / K, 0.05 / K
# GFL (วงควบคุมเขียนเอง) PLL เท่าตัวอย่าง DPsim, วงกำลังตั้งเป็นแบนด์วิดท์, วงกระแส 500 Hz
GFL = dict(KP_PLL=0.25, KI_PLL=0.2, KPP_N=0.2, KIP_N=30.0, FC=500.0, WFF=2 * np.pi * 50.0, WC=W)
# GFM (SSN_GFM จูนใหม่ให้ต่อกริดได้)
GFM = dict(J=0.2 * K, D=25.0 * K, PF_CUT=30.0, QDROOP=0.05, QCUT=50.0, KPV=0.05 * K, WV=800.0, WI=3000.0,
           FF=0.8, DELAY=20e3 / 1.5, KU=1 / 15, KQ=0.3 / K)
ZB = VLV ** 2 / S
M3 = dp.Math.single_phase_parameter_to_three_phase
LOGS = Path("logs_dpsim")


# =====================================================================
# 1) สร้างระบบใน DPsim
# =====================================================================
def make_gfm(xv):
    c = GFM
    inv = dp.emt.ph3.SSN_GFM("GFM", "GFM")
    inv.set_parameters(LF, CF, RF, RC, VPK, W, P_SET, 0.0, c["J"], c["D"], c["KU"], c["KQ"],
                       c["KPV"], c["WV"] ** 2 * CF, c["WI"] * LF, c["WI"] * RF, 0.0, c["PF_CUT"], c["DELAY"])
    inv.set_reactive_power_droop(c["QDROOP"] * VPK / S, c["QCUT"])
    inv.set_grid_current_feedforward(c["FF"])
    if xv > 0:
        inv.set_virtual_impedance(xv * ZB / 5, xv * ZB)
    inv.set_numerical_linearization_parameters(1e-6, 1e-8)
    return inv, "omega_gfm", ("i_grid_d", "i_grid_q")


def build_gfm(kind, scr, pll=1.0, xv=0.0, t_step=T_STEP, t_end=T_END):
    name = f"{kind}_scr{scr}_pll{pll}_xv{xv}".replace(".", "p")
    root = LOGS / name; root.mkdir(parents=True, exist_ok=True)
    rg = ZB / scr / np.sqrt(1 + XR ** 2); lg = rg * XR / W
    rt = ZT * ZB / np.sqrt(1 + XRT ** 2); lt = rt * XRT / W

    # power flow ให้จุดทำงานเริ่มต้น
    dp.Logger.set_log_dir(str(root / "pf"))
    nG = dp.sp.SimNode("nGrid", dp.PhaseType.Single); nP = dp.sp.SimNode("nPcc", dp.PhaseType.Single)
    nL = dp.sp.SimNode("nLV", dp.PhaseType.Single)
    sl = dp.sp.ph1.NetworkInjection("Slack", dp.LogLevel.off)
    sl.set_parameters(voltage_set_point=VLV); sl.set_base_voltage(VLV); sl.modify_power_flow_bus_type(dp.PowerflowBusType.VD)
    ln = dp.sp.ph1.PiLine("Grid", dp.LogLevel.off); ln.set_parameters(R=rg, L=lg, C=0.0, G=0.0); ln.set_base_voltage(VLV)
    tz = dp.sp.ph1.PiLine("Trafo", dp.LogLevel.off); tz.set_parameters(R=rt, L=lt, C=0.0, G=0.0); tz.set_base_voltage(VLV)
    ld = dp.sp.ph1.Load("Inv", dp.LogLevel.off)
    ld.set_parameters(active_power=-P_SET, reactive_power=0.0, nominal_voltage=VLV)
    ld.modify_power_flow_bus_type(dp.PowerflowBusType.PQ)
    sl.connect([nG]); ln.connect([nP, nG]); tz.connect([nL, nP]); ld.connect([nL])
    spf = dp.SystemTopology(F, [nG, nP, nL], [sl, ln, tz, ld])
    pf = dp.Simulation(name + "_pf", dp.LogLevel.off); pf.set_system(spf); pf.set_time_step(1.0); pf.set_final_time(1.0)
    pf.set_domain(dp.Domain.SP); pf.set_solver(dp.Solver.NRP)
    pf.set_solver_component_behaviour(dp.SolverBehaviour.Initialization); pf.do_init_from_nodes_and_terminals(False)
    pf.run()

    # EMT
    dp.Logger.set_log_dir(str(root))
    eG = dp.emt.SimNode("nGrid", dp.PhaseType.ABC); eP = dp.emt.SimNode("nPcc", dp.PhaseType.ABC)
    eL = dp.emt.SimNode("nLV", dp.PhaseType.ABC); eD = dp.emt.SimNode("nLoad", dp.PhaseType.ABC)
    esl = dp.emt.ph3.NetworkInjection("Slack", dp.LogLevel.off)
    eln = dp.emt.ph3.PiLine("Grid", dp.LogLevel.off); eln.set_parameters(M3(rg), M3(lg), M3(0.0), M3(0.0))
    etz = dp.emt.ph3.PiLine("Trafo", dp.LogLevel.off); etz.set_parameters(M3(rt), M3(lt), M3(0.0), M3(0.0))
    inv, w_attr, ia = make_gfm(xv)
    br = dp.emt.ph3.SeriesSwitch("Br", dp.LogLevel.off); br.set_parameters(1e9, 1e-4, False); br.open()
    rl = dp.emt.ph3.SeriesResistor("Load", dp.LogLevel.off); rl.set_parameters(VLV ** 2 / LOAD)
    esl.connect([eG]); eln.connect([eP, eG]); etz.connect([eL, eP])
    inv.connect([dp.emt.SimNode.gnd, eL]); br.connect([eD, eP]); rl.connect([dp.emt.SimNode.gnd, eD])
    sys_ = dp.SystemTopology(F, [eG, eP, eL, eD], [esl, eln, etz, inv, br, rl])
    sys_.init_with_powerflow(spf, dp.Domain.EMT)
    log = dp.Logger(name)
    log.log_attribute("p", "p_inst", inv); log.log_attribute("w", w_attr, inv)
    log.log_attribute("id", ia[0], inv); log.log_attribute("iq", ia[1], inv)
    sim = dp.Simulation(name, dp.LogLevel.off); sim.set_system(sys_); sim.add_logger(log)
    sim.set_domain(dp.Domain.EMT); sim.set_solver(dp.Solver.MNA)
    sim.do_system_matrix_recomputation(True); sim.do_init_from_nodes_and_terminals(True)
    sim.set_time_step(DT); sim.set_final_time(t_end)
    sim.add_event(dp.event.SwitchEvent(t_step, br, True))
    return sim, root / f"{name}.csv", esl


def read(kind, csv):
    df = pd.read_csv(csv, skipinitialspace=True); df.columns = [c.strip() for c in df.columns]
    t = df["time"].to_numpy(); p = df["p"].to_numpy() / S
    i = np.hypot(df["id"].to_numpy(), df["iq"].to_numpy()) / (np.sqrt(2) * S / (np.sqrt(3) * VLV))
    return dict(t=t, p=p, f=df["w"].to_numpy() / (2 * np.pi), i=i)


# ---------------------------------------------------------------- GFL: โครงข่ายใน DPsim + วงควบคุมใน Python
K3 = np.sqrt(1.5)


def park(x, th):
    a = np.array([np.cos(th), np.cos(th - 2 * np.pi / 3), np.cos(th + 2 * np.pi / 3)])
    b = -np.array([np.sin(th), np.sin(th - 2 * np.pi / 3), np.sin(th + 2 * np.pi / 3)])
    return np.sqrt(2 / 3) * (a @ x), np.sqrt(2 / 3) * (b @ x)          # power-invariant


def ipark(d, q, th):
    a = np.array([np.cos(th), np.cos(th - 2 * np.pi / 3), np.cos(th + 2 * np.pi / 3)])
    b = -np.array([np.sin(th), np.sin(th - 2 * np.pi / 3), np.sin(th + 2 * np.pi / 3)])
    return np.sqrt(2 / 3) * (a * d + b * q)


def simulate_gfl(scr, pll=1.0, t_load=T_STEP, t_end=T_END, jump=0.0, t_jump=1.0):
    """กริด -> Zg -> บัสจุดต่อ (โหลด) -> Zt -> Rc -> Cf -> Rf+Lf -> แหล่งแรงดันที่ตัวควบคุมสั่ง"""
    rg = ZB / scr / np.sqrt(1 + XR ** 2); lg = rg * XR / W
    rt = ZT * ZB / np.sqrt(1 + XRT ** 2); lt = rt * XRT / W
    # จุดทำงานเริ่มต้นด้วยเฟสเซอร์ (ค่ายอด) ให้ Q ที่ตัวเก็บประจุเป็นศูนย์
    vg = complex(VPK); z = RC + rg + rt + 1j * W * (lg + lt); vc = vg
    for _ in range(300):
        ig = np.conj(P_SET / (1.5 * vc)); vc = vg + z * ig
    i_f = ig + 1j * W * CF * vc; vconv = vc + (RF + 1j * W * LF) * i_f
    vpcc = vg + (rg + 1j * W * lg) * ig; vlv = vpcc + (rt + 1j * W * lt) * ig

    name = f"gfl_scr{scr}_pll{pll}_j{jump}".replace(".", "p").replace("-", "m")
    root = LOGS / name; root.mkdir(parents=True, exist_ok=True); dp.Logger.set_log_dir(str(root))
    a3 = np.array([1, np.exp(-2j * np.pi / 3), np.exp(2j * np.pi / 3)]).reshape(3, 1)
    nodes = {}
    for k_, v_ in (("nGrid", vg), ("nPcc", vpcc), ("nLV", vlv), ("nC", vc), ("nV", vconv), ("nLoad", 0j)):
        nodes[k_] = dp.emt.SimNode(k_, dp.PhaseType.ABC)
        nodes[k_].set_initial_voltage(v_ * a3 * np.sqrt(1.5))
    esl = dp.emt.ph3.NetworkInjection("Slack", dp.LogLevel.off); esl.set_parameters((VLV * a3).astype(complex), F)
    eln = dp.emt.ph3.PiLine("Grid", dp.LogLevel.off); eln.set_parameters(M3(rg), M3(lg), M3(0.0), M3(0.0))
    etz = dp.emt.ph3.PiLine("Trafo", dp.LogLevel.off); etz.set_parameters(M3(rt), M3(lt), M3(0.0), M3(0.0))
    erc = dp.emt.ph3.Resistor("Rc", dp.LogLevel.off); erc.set_parameters(M3(RC))
    ecf = dp.emt.ph3.Capacitor("Cf", dp.LogLevel.off); ecf.set_parameters(M3(CF))
    elf = dp.emt.ph3.PiLine("Lf", dp.LogLevel.off); elf.set_parameters(M3(RF), M3(LF), M3(0.0), M3(0.0))
    evs = dp.emt.ph3.ControlledVoltageSource("Vconv", dp.LogLevel.off)
    evs.set_parameters((abs(vconv) * np.cos(np.angle(vconv) + np.angle(a3.flatten()))).reshape(3, 1))
    br = dp.emt.ph3.SeriesSwitch("Br", dp.LogLevel.off); br.set_parameters(1e9, 1e-4, False); br.open()
    rl = dp.emt.ph3.SeriesResistor("Load", dp.LogLevel.off); rl.set_parameters(VLV ** 2 / LOAD)
    G = dp.emt.SimNode.gnd; n = nodes
    esl.connect([n["nGrid"]]); eln.connect([n["nPcc"], n["nGrid"]]); etz.connect([n["nLV"], n["nPcc"]])
    erc.connect([n["nLV"], n["nC"]]); ecf.connect([G, n["nC"]]); elf.connect([n["nC"], n["nV"]])
    evs.connect([G, n["nV"]]); br.connect([n["nLoad"], n["nPcc"]]); rl.connect([G, n["nLoad"]])
    sim = dp.Simulation(name, dp.LogLevel.off)
    sim.set_system(dp.SystemTopology(F, list(n.values()), [esl, eln, etz, erc, ecf, elf, evs, br, rl]))
    sim.set_domain(dp.Domain.EMT); sim.set_solver(dp.Solver.MNA); sim.do_system_matrix_recomputation(True)
    sim.do_init_from_nodes_and_terminals(True); sim.set_time_step(DT); sim.set_final_time(t_end)
    sim.add_event(dp.event.SwitchEvent(t_load, br, True))

    # ---- ตัวควบคุม
    c = GFL; h = DT
    kp_pll, ki_pll = c["KP_PLL"] * pll, c["KI_PLL"] * pll ** 2
    vd0 = K3 * VPK; kpp, kip = c["KPP_N"] / vd0, c["KIP_N"] / vd0
    wc = 2 * np.pi * c["FC"]; kpc, kic = wc * LF, 5 * wc * RF
    th = np.angle(vc); igl0 = K3 * ig * np.exp(-1j * th)
    phi, Pf, Qf, xd, xq, gd, gq = 0.0, P_SET, 0.0, igl0.real / kip, igl0.imag / kip, 0.0, 0.0
    vfd, vfq = K3 * abs(vc), 0.0
    vref, vsrc = evs.attr("V_ref"), esl.attr("V_ref")
    vcn, vlvn, ilf = n["nC"].attr("v"), n["nLV"].attr("v"), elf.attr("i_intf")
    rec = {k_: [] for k_ in ("t", "p", "f", "i")}
    old_err = np.seterr(all="ignore")                                     # กรณีหลุด ค่าจะล้น ไม่ต้องเตือน
    sim.start(); v0 = vsrc.get(); t = 0.0; k = 0; jumped = False; sgn = None
    while t < t_end - 1e-9:
        t = sim.next(); k += 1
        if jump and not jumped and t >= t_jump:
            vsrc.set(v0 * np.exp(1j * np.deg2rad(jump))); jumped = True
        if k == 1:
            th = np.angle(vc) + W * (t - h)                               # ค่าที่อ่านหลัง sim.next() เป็นของเวลา t - h
        vabc = np.asarray(vcn.get()).reshape(3)
        # i_intf ของ Resistor ไม่อัปเดตระหว่างจำลอง จึงคำนวณกระแสกริดจากแรงดันคร่อม Rc
        ig_abc = (vabc - np.asarray(vlvn.get()).reshape(3)) / RC; if_abc = np.asarray(ilf.get()).reshape(3)
        if sgn is None:
            sgn = (np.sign(vabc @ ig_abc) or 1.0, np.sign(vabc @ if_abc) or 1.0)
        ig_abc, if_abc = sgn[0] * ig_abc, sgn[1] * if_abc
        vd, vq = park(vabc, th); igd, igq = park(ig_abc, th); ifd, ifq = park(if_abc, th)
        w = W + kp_pll * vq + ki_pll * phi; phi += vq * h                 # PLL
        p = vd * igd + vq * igq; q = vq * igd - vd * igq
        Pf += c["WC"] * (p - Pf) * h; Qf += c["WC"] * (q - Qf) * h
        xd += (P_SET - Pf) * h; xq += Qf * h
        igd_ref = kpp * (P_SET - Pf) + kip * xd; igq_ref = kpp * Qf + kip * xq   # วงกำลัง
        ed = igd_ref - w * CF * vq - ifd; eq = igq_ref + w * CF * vd - ifq      # อ้างอิงกระแสฝั่งคอนเวอร์เตอร์
        gd += ed * h; gq += eq * h
        vfd += c["WFF"] * (vd - vfd) * h; vfq += c["WFF"] * (vq - vfq) * h     # feedforward ผ่านตัวกรอง
        ud = vfd - w * LF * ifq + kpc * ed + kic * gd
        uq = vfq + w * LF * ifd + kpc * eq + kic * gq
        th += w * h
        vref.set(ipark(ud, uq, th).reshape(3, 1))
        if k % 2 == 0:
            rec["t"].append(t); rec["p"].append(p / S); rec["f"].append(w / (2 * np.pi))
            rec["i"].append(np.hypot(igd, igq) / (S / VLV))
    sim.stop()
    np.seterr(**old_err)
    return {k_: np.array(v_) for k_, v_ in rec.items()}


def classify(r):
    t, p = r["t"], r["p"]
    if not (np.all(np.isfinite(p)) and np.nanmax(np.abs(p)) < 10):
        return "หลุด"
    early = np.ptp(p[(t > 1.05) & (t < 1.55)]); late = np.ptp(p[t > 2.5])
    return "แกว่งไม่หาย" if (late > 0.3 * early and late > 1e-3) else "นิ่ง"


def run_step(kind, scr, pll=1.0, xv=0.0):
    """ต่อโหลดที่ T_STEP แล้วจัดกลุ่มผล: นิ่ง / แกว่งไม่หาย / หลุด"""
    if kind == "gfl":
        r = simulate_gfl(scr, pll)
    else:
        sim, csv, _ = build_gfm("gfm", scr, xv=xv)
        sim.run()
        r = read("gfm", csv)
    r["state"] = classify(r)
    return r


def run_jump(kind, scr, pll=1.0, xv=0.0, jump=JUMP_DEG, t_end=2.0):
    """หมุนมุมแรงดันของแหล่งจ่ายกริดที่ t = 1 s ระหว่างการจำลอง"""
    if kind == "gfl":
        return simulate_gfl(scr, pll, t_load=99.0, t_end=t_end, jump=jump)
    sim, csv, esl = build_gfm("gfm", scr, xv=xv, t_step=99.0, t_end=t_end)
    vref = esl.attr("V_ref")
    sim.start(); v0 = vref.get(); t = 0.0; done = False
    while t < t_end - 1e-9:
        t = sim.next()
        if not done and t >= 1.0:
            vref.set(v0 * np.exp(1j * np.deg2rad(jump))); done = True
    sim.stop()
    return read("gfm", csv)


# =====================================================================
# 2) ขอบเสถียรภาพ
#    GFM: สมการชุดเดียวกับในซอร์สโค้ดของ DPsim แล้วหา eigenvalue (DPsim ไม่มีให้)
#    GFL: ตัวควบคุมเขียนเอง จึงหาขอบด้วยการรัน DPsim ไล่ SCR แบบ bisection
#    เวกเตอร์เชิงซ้อนในเฟรมหมุนที่ 50 Hz วงจร: vc --(Rc + Zt + Zg)--> กริด
# =====================================================================
def grid_rl(scr):
    rg = ZB / scr / np.sqrt(1 + XR ** 2); rt = ZT * ZB / np.sqrt(1 + XRT ** 2)
    return RC + rg + rt, (rg * XR + rt * XRT) / W


def gfm_f(x, scr, xv, Eset):
    c = GFM; R, L = grid_rl(scr)
    Pf, Qf, w, d, E, xvd, xvq, xid, xiq, ud, uq = x[:11]
    vc, i_f, ig = x[11] + 1j * x[12], x[13] + 1j * x[14], x[15] + 1j * x[16]
    rot = np.exp(-1j * d); vcl, ifl, igl = vc * rot, i_f * rot, ig * rot
    s_ = 1.5 * vc * np.conj(ig)
    ev = E - (xv * ZB / 5 + 1j * xv * ZB) * ifl - vcl
    iref = c["FF"] * igl + 1j * w * CF * vcl + c["KPV"] * ev + c["WV"] ** 2 * CF * (xvd + 1j * xvq)
    ei = iref - ifl
    vconv = vcl + 1j * w * LF * ifl + c["WI"] * LF * ei + c["WI"] * RF * (xid + 1j * xiq)
    du = c["DELAY"] * (vconv - (ud + 1j * uq))
    vinv = (ud + 1j * uq) / rot
    dvc = (i_f - ig) / CF - 1j * W * vc
    dif = (vinv - vc - RF * i_f) / LF - 1j * W * i_f
    dig = (vc - VPK - R * ig) / L - 1j * W * ig
    dq = c["QDROOP"] * VPK / S
    return np.array([c["PF_CUT"] * (s_.real - Pf), c["PF_CUT"] * (s_.imag - Qf),
                     ((P_SET - Pf) / w - c["D"] * (w - W)) / c["J"], w - W, c["QCUT"] * (Eset - dq * Qf - E),
                     ev.real, ev.imag, ei.real, ei.imag, du.real, du.imag,
                     dvc.real, dvc.imag, dif.real, dif.imag, dig.real, dig.imag])


def operating_point(scr, xv=0.0):
    R, L = grid_rl(scr); z = R + 1j * W * L
    vc = complex(VPK)
    for _ in range(200):
        ig = np.conj(P_SET / (1.5 * vc)); vc = VPK + z * ig
    i_f = ig + 1j * W * CF * vc
    vinv = vc + (RF + 1j * W * LF) * i_f
    return vc, i_f, ig, vinv


def eigen_gfm(scr, xv=0.0):
    vc, i_f, ig, vinv = operating_point(scr, xv)
    c = GFM; E0 = vc + (xv * ZB / 5 + 1j * xv * ZB) * i_f; d0 = np.angle(E0); rot = np.exp(-1j * d0)
    vcl, ifl, igl, vil = vc * rot, i_f * rot, ig * rot, vinv * rot
    ev = abs(E0) - (xv * ZB / 5 + 1j * xv * ZB) * ifl - vcl
    xvv = (ifl - c["FF"] * igl - 1j * W * CF * vcl - c["KPV"] * ev) / (c["WV"] ** 2 * CF)
    xii = (vil - vcl - 1j * W * LF * ifl) / (c["WI"] * RF)
    x0 = np.array([P_SET, 0, W, d0, abs(E0), xvv.real, xvv.imag, xii.real, xii.imag, vil.real, vil.imag,
                   vc.real, vc.imag, i_f.real, i_f.imag, ig.real, ig.imag])
    fun = lambda x: gfm_f(x, scr, xv, abs(E0))
    x = fsolve(fun, x0, xtol=1e-12)
    J = np.zeros((len(x), len(x)))
    for j in range(len(x)):
        h = 1e-6 * max(1.0, abs(x[j])); xp, xm = x.copy(), x.copy(); xp[j] += h; xm[j] -= h
        J[:, j] = (fun(xp) - fun(xm)) / (2 * h)
    return np.linalg.eigvals(J)


def unstable_gfm(scr, xv=0.0):
    lam = eigen_gfm(scr, xv)
    return lam[np.abs(lam.imag) < 2 * np.pi * 300].real.max() > 0      # ดูเฉพาะโหมดต่ำกว่า 300 Hz


# ขอบของ GFL ที่ไล่ไว้ตอนทำโพสต์ (ความเร็ว PLL: SCR ที่ต่ำกว่านี้ไม่เสถียร) ใช้ตอน QUICK = True จะได้ไม่ต้องรอ
GFL_BOUNDARY_POST = {2: 1.84, 3: 2.07, 4: 2.24, 5: 2.42, 6: 2.73, 7: 3.07, 8: 3.59, 9: 4.04, 10: 4.55, 11: 5.13, 12: 5.55}


def boundary_gfm(values, lo=1.95, hi=1000.0):
    """GFM ไม่เสถียรเมื่อ SCR สูงกว่าขอบ หาจาก eigenvalue"""
    out = []
    for v in values:
        if not unstable_gfm(hi, v):
            out.append(None); continue
        a, b = lo, hi
        for _ in range(30):
            m = np.sqrt(a * b)
            if unstable_gfm(m, v):
                b = m
            else:
                a = m
        out.append(np.sqrt(a * b))
    return out


def boundary_gfl(values, lo=1.6, hi=20.0, steps=6):
    """GFL ไม่เสถียรเมื่อ SCR ต่ำกว่าขอบ หาจากการรัน DPsim 2 วินาทีแบบ bisection (ช้า รันละหลายวินาที)"""
    def stable(scr, pll):
        r = simulate_gfl(scr, pll, t_load=99.0, t_end=2.0)
        p = r["p"]
        return np.all(np.isfinite(p)) and np.nanmax(np.abs(p)) < 5 and np.ptp(p[r["t"] > 1.5]) < 2e-3
    out = []
    for v in values:
        if stable(lo, v):
            out.append(None); continue
        a, b = lo, hi
        for _ in range(steps):
            m = np.sqrt(a * b)
            if stable(m, v):
                b = m
            else:
                a = m
        out.append(np.sqrt(a * b))
    return out


# =====================================================================
# 2.1) damping ratio
#      GFM ใช้ eigen_gfm ข้างบน ส่วน GFL เขียนสมการของตัวควบคุมใน simulate_gfl ขึ้นมาใหม่
#      การหน่วงหนึ่ง time step ของตัวควบคุมประมาณเป็นตัวกรองอันดับหนึ่ง 1.5 DT
#      ตรวจแล้ว SCR ที่ damping ข้ามศูนย์ต่างจากที่ไล่ใน DPsim ไม่เกินราว 3%
# =====================================================================
def gfl_f(x, scr, pll):
    c = GFL; R, L = grid_rl(scr); tau = 1.5 * DT
    kp_pll, ki_pll = c["KP_PLL"] * pll, c["KI_PLL"] * pll ** 2
    vd0 = K3 * VPK; kpp, kip = c["KPP_N"] / vd0, c["KIP_N"] / vd0
    wc = 2 * np.pi * c["FC"]; kpc, kic = wc * LF, 5 * wc * RF
    d, phi, Pf, Qf, xd, xq, gd, gq, vfd, vfq, ud, uq = x[:12]
    vc, i_f, ig = x[12] + 1j * x[13], x[14] + 1j * x[15], x[16] + 1j * x[17]
    rot = np.exp(-1j * d); vcl, ifl, igl = K3 * vc * rot, K3 * i_f * rot, K3 * ig * rot
    s_ = vcl * np.conj(igl)
    dd = kp_pll * vcl.imag + ki_pll * phi; w = W + dd
    iref = kpp * (P_SET - Pf) + kip * xd + 1j * (kpp * Qf + kip * xq)
    e = iref + 1j * w * CF * vcl - ifl
    ucmd = vfd + 1j * vfq + 1j * w * LF * ifl + kpc * e + kic * (gd + 1j * gq)
    du = (ucmd - (ud + 1j * uq)) / tau
    vinv = (ud + 1j * uq) / rot / K3
    dvc = (i_f - ig) / CF - 1j * W * vc
    dif = (vinv - vc - RF * i_f) / LF - 1j * W * i_f
    dig = (vc - VPK - R * ig) / L - 1j * W * ig
    return np.array([dd, vcl.imag, c["WC"] * (s_.real - Pf), c["WC"] * (s_.imag - Qf), P_SET - Pf, Qf,
                     e.real, e.imag, c["WFF"] * (vcl.real - vfd), c["WFF"] * (vcl.imag - vfq), du.real, du.imag,
                     dvc.real, dvc.imag, dif.real, dif.imag, dig.real, dig.imag])


def eigen_gfl(scr, pll=1.0):
    c = GFL; vc, i_f, ig, vinv = operating_point(scr)
    vd0 = K3 * VPK; kip = c["KIP_N"] / vd0; kic = 5 * 2 * np.pi * c["FC"] * RF
    d0 = np.angle(vc); rot = np.exp(-1j * d0)
    vcl, ifl, igl, vil = K3 * vc * rot, K3 * i_f * rot, K3 * ig * rot, K3 * vinv * rot
    g = (vil - vcl - 1j * W * LF * ifl) / kic
    x0 = np.array([d0, 0, P_SET, 0, igl.real / kip, igl.imag / kip, g.real, g.imag, vcl.real, vcl.imag,
                   vil.real, vil.imag, vc.real, vc.imag, i_f.real, i_f.imag, ig.real, ig.imag])
    fun = lambda x: gfl_f(x, scr, pll)
    x = fsolve(fun, x0, xtol=1e-12)
    J = np.zeros((len(x), len(x)))
    for j in range(len(x)):
        h = 1e-6 * max(1.0, abs(x[j])); xp, xm = x.copy(), x.copy(); xp[j] += h; xm[j] -= h
        J[:, j] = (fun(xp) - fun(xm)) / (2 * h)
    return np.linalg.eigvals(J)


def damping_ratio(lam, fmax=400.0):
    """damping ratio ของโหมดที่หน่วงน้อยที่สุด ในช่วงความถี่ต่ำกว่า fmax Hz"""
    o = lam[(lam.imag > 1) & (lam.imag < 2 * np.pi * fmax)]
    return float(np.min(-o.real / np.abs(o)))

# =====================================================================
# 3) สไตล์ของรูป (ชุดเดียวกับรูปที่ใช้ในโพสต์)
# =====================================================================
OUTDIR = Path(".")                    # โฟลเดอร์ที่เซฟรูป
SURF, INK, INK2, MUTED = "#fcfcfb", "#1d1d1b", "#52514e", "#898781"
GRID_C, AXIS, RULE = "#e1e0d9", "#c3c2b7", "#e1e0d9"
BLUE, BLUE2, BLUE3, BLUE4 = "#1f5fae", "#2a78d6", "#6fa6e6", "#a9c9f0"
ORANGE, ORANGE2, ORANGE3 = "#c4501f", "#eb6834", "#f3a37f"
FONT_REG = FONT_BOLD = None


def thai_font():
    """ใช้ฟอนต์ไทยถ้ามี (บน Colab ติดตั้งด้วย !apt-get install -y fonts-thai-tlwg)"""
    global FONT_REG, FONT_BOLD
    from matplotlib import font_manager
    d = Path("/usr/share/fonts/truetype/tlwg")
    if (d / "Garuda.ttf").exists():
        for f in ("Garuda.ttf", "Garuda-Bold.ttf"):
            if (d / f).exists():
                font_manager.fontManager.addfont(str(d / f))
        FONT_REG = font_manager.FontProperties(fname=str(d / "Garuda.ttf"))
        if (d / "Garuda-Bold.ttf").exists():
            FONT_BOLD = font_manager.FontProperties(fname=str(d / "Garuda-Bold.ttf"))
        plt.rcParams["font.family"] = [FONT_REG.get_name(), "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams.update({
        "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID_C, "grid.linewidth": 0.8, "xtick.color": MUTED, "ytick.color": MUTED,
        "text.color": INK, "legend.frameon": False, "legend.labelcolor": INK2,
    })


thai_font()


def title(fig, main, sub):
    fig.text(0.02, 0.975, main, fontsize=14, weight="bold", va="top")
    fig.text(0.02, 0.918, sub, fontsize=10, color=INK2, va="top")


def emit(fig, name, crop_bottom=None):
    """เซฟรูปเป็นไฟล์ png แล้วแสดงในโน้ตบุ๊ก (ไฟล์ที่ได้คือรูปเดียวกับที่ใช้ในโพสต์)"""
    OUTDIR.mkdir(parents=True, exist_ok=True)
    path = OUTDIR / f"{name}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    if crop_bottom:
        try:
            from PIL import Image
            im = Image.open(path)
            im.crop((0, 0, im.width, min(int(crop_bottom * im.height), im.height))).save(path)
        except ImportError:
            pass
    try:
        from IPython.display import Image as IImage, display
        display(IImage(str(path)))
    except ImportError:
        pass
    print("บันทึก", path)
    return path


def swing(t, y, win):
    """ขนาดการแกว่งจากยอดถึงยอด แบบเลื่อนหน้าต่างและดูย้อนหลังอย่างเดียว
    (GFL แกว่ง 80-90 Hz วาดเส้นดิบแล้วทึบจนอ่านไม่ออก)"""
    n = max(3, int(round(win / (t[1] - t[0])))); ker = np.ones(n) / n
    avg = np.convolve(np.pad(y, (n - 1, 0), mode="edge"), ker, mode="valid")
    x = y - avg
    ms = np.convolve(np.pad(x * x, (n - 1, 0), mode="edge"), ker, mode="valid")
    return 2 * np.sqrt(2) * np.sqrt(ms)


def cut_at_blowup(t, p):
    """ตัดหางที่ค่าระเบิด คืน (t, p, เวลาที่หลุด หรือ None)"""
    bad = np.where(~np.isfinite(p) | (np.abs(p) > 3))[0]
    tb = t[bad[0]] if len(bad) else None
    k = t < (tb - 0.005 if tb else 9e9)
    return t[k], p[k], tb


# =====================================================================
# 4) รันเก็บผลไว้ใช้ซ้ำ (แต่ละรูปเรียกใช้ของชุดเดียวกัน ไม่ต้องรันซ้ำ)
# =====================================================================
CACHE = {}


def _point(kind, scr, pll=1.0, xv=0.0):
    r = run_step(kind, scr, pll=pll, xv=xv)
    t, p = r["t"], r["p"]
    fin = bool(np.all(np.isfinite(p)) and np.nanmax(np.abs(p)) < 10)
    pp = lambda m_: float(np.nanmax(p[m_]) - np.nanmin(p[m_])) if fin else np.nan
    print(f"  {kind} SCR {scr} " + (f"PLL x{pll}" if kind == "gfl" else f"xv {xv}") + f": {r['state']}", flush=True)
    return dict(kind=kind, scr=scr, pll=pll, xv=xv, state=r["state"], finite=fin,
                pp_early=pp((t > 1.05) & (t < 1.55)), pp_late=pp(t > 2.5), t=t[::4], p=p[::4])


def map_gfl():
    """ผลของแผนที่ฝั่ง GFL (ต่อโหลดที่ 1 วินาที ทุกคู่ SCR กับความเร็ว PLL)"""
    if "gfl" not in CACHE:
        rows = PLL_LIST if not QUICK else QUICK_PLL
        print(f"รัน GFL {len(rows) * len(SCR_LIST)} กรณี")
        CACHE["gfl"] = [_point("gfl", s, pll=v) for v in rows for s in SCR_LIST]
    return CACHE["gfl"]


def map_gfm():
    """ผลของแผนที่ฝั่ง GFM"""
    if "gfm" not in CACHE:
        rows = XV_LIST if not QUICK else QUICK_XV
        print(f"รัน GFM {len(rows) * len(SCR_LIST)} กรณี")
        CACHE["gfm"] = [_point("gfm", s, xv=v) for v in rows for s in SCR_LIST]
    return CACHE["gfm"]


def pick(rows, **kw):
    for r in rows:
        if all(abs(r[k] - v) < 1e-9 for k, v in kw.items()):
            return r
    return None


def jump(kind, scr, pll=1.0, xv=0.0):
    key = ("pj", kind, scr, pll, xv)
    if key not in CACHE:
        print(f"  มุมกระโดด {kind} SCR {scr} " + (f"PLL x{pll}" if kind == "gfl" else f"xv {xv}"), flush=True)
        CACHE[key] = run_jump(kind, scr, pll=pll, xv=xv)
    return CACHE[key]


def zeta_curves():
    """damping ratio จาก eigenvalue ตาม SCR (คำนวณเร็ว ไม่ต้องรัน DPsim)"""
    if "zeta" not in CACHE:
        scr = np.logspace(np.log10(1.75), 3, 120)
        z = {"scr": scr}
        for v in PLL_LIST_Z:
            z[f"gfl{v}"] = np.array([damping_ratio(eigen_gfl(s, v)) for s in scr])
        for v in XV_LIST:
            z[f"gfm{v}"] = np.array([damping_ratio(eigen_gfm(s, v)) for s in scr])
        CACHE["zeta"] = z
    return CACHE["zeta"]


def boundary_gfl_curve():
    """ขอบของ GFL ไล่ SCR แบบ bisection ใน DPsim (ราว 2 นาที) หรือใช้ค่าที่ไล่ไว้ตอนทำโพสต์เมื่อ QUICK"""
    if "bgfl" not in CACHE:
        v = list(GFL_BOUNDARY_POST)
        CACHE["bgfl"] = (v, [GFL_BOUNDARY_POST[k] for k in v]) if QUICK else (v, boundary_gfl(v))
    return CACHE["bgfl"]


def scr_axis(ax, smax=1000):
    ax.set_xscale("log"); ax.set_xlim(smax * 1.3, 1.6)      # กริดแข็งอยู่ซ้าย อ่อนลงไปทางขวา
    xt = [2, 5, 10, 20, 50, 100, 300, 1000]
    ax.set_xticks(xt); ax.set_xticklabels([str(x) for x in xt]); ax.minorticks_off()
    ax.set_xlabel("SCR ที่บัส 22 kV (ยิ่งไปทางขวากริดยิ่งอ่อน)")


STYLE = {"นิ่ง": dict(marker="o", facecolors="none", edgecolors=BLUE2, s=60, lw=1.6),
         "แกว่งไม่หาย": dict(marker="o", facecolors=MUTED, edgecolors=MUTED, s=60, lw=1.2),
         "หลุด": dict(marker="o", facecolors=ORANGE2, edgecolors=ORANGE2, s=60, lw=1.2)}


# =====================================================================
# 5) รูปที่ใช้ในโพสต์
# =====================================================================
def fig1_system():
    """รูป 1 ผังระบบที่จำลอง (วาดอย่างเดียว ไม่ต้องรัน)"""
    from matplotlib.patches import Circle, FancyBboxPatch, Rectangle, Polygon
    fig = plt.figure(figsize=(11, 6.2))
    ax = fig.add_axes([0.0, 0.0, 1.0, 0.86])
    ax.set_xlim(0, 11); ax.set_ylim(0, 6.2); ax.axis("off"); ax.grid(False)
    LW = dict(color=INK, lw=1.8, solid_capstyle="round")
    YM = 3.9

    def line(xs_, ys_, **kw):
        k = dict(LW); k.update(kw); ax.plot(xs_, ys_, **k)

    def bus(x, name, sub):
        ax.plot([x, x], [YM - 1.1, YM + 1.1], color=INK, lw=6, solid_capstyle="butt")
        ax.text(x, YM + 1.62, name, ha="center", fontsize=12, weight="bold")
        ax.text(x, YM + 1.25, sub, ha="center", va="bottom", fontsize=9, color=MUTED)

    xs = 0.75
    ax.add_patch(Circle((xs, YM), 0.45, fill=False, ec=INK, lw=1.8))
    tt = np.linspace(-0.27, 0.27, 60); ax.plot(xs + tt, YM + 0.12 * np.sin(tt / 0.27 * np.pi), color=INK, lw=1.6)
    ax.text(xs, YM - 0.75, "แหล่งจ่ายกริด", ha="center", fontsize=11, weight="bold")
    ax.text(xs, YM - 1.07, "22 kV 50 Hz", ha="center", fontsize=9.5, color=INK2)
    ax.text(xs, YM - 1.35, "NetworkInjection", ha="center", fontsize=8.5, color=MUTED)
    ax.text(xs, YM + 0.7, f"มุมเฟสกระโดด {abs(JUMP_DEG):g}°", ha="center", fontsize=9.5, color=ORANGE2)

    xb1 = 1.85
    line([xs + 0.45, xb1], [YM, YM]); bus(xb1, "บัสกริด 22 kV", "nGrid")

    xz0, xz1 = 2.4, 3.6
    line([xb1, xz0], [YM, YM])
    xx = np.linspace(xz0, xz0 + 0.4, 9)
    ax.plot(xx, YM + 0.12 * np.array([0, 1, -1, 1, -1, 1, -1, 1, 0]), **LW)
    xl = np.linspace(xz0 + 0.4, xz1, 200)
    ax.plot(xl, YM + 0.15 * np.abs(np.sin((xl - xz0 - 0.4) / (xz1 - xz0 - 0.4) * 3 * np.pi)), **LW)
    ax.text((xz0 + xz1) / 2, YM + 0.45, "อิมพีแดนซ์กริด", ha="center", fontsize=10.5, weight="bold")
    ax.text((xz0 + xz1) / 2, YM - 0.35, "ปรับตาม SCR", ha="center", va="top", fontsize=9.5, color=INK2)
    ax.text((xz0 + xz1) / 2, YM - 0.65, f"{min(SCR_LIST):g} ถึง {max(SCR_LIST):g}", ha="center", va="top", fontsize=9.5, color=INK2)
    ax.text((xz0 + xz1) / 2, YM - 0.95, f"X/R {XR:g}  PiLine", ha="center", va="top", fontsize=8.5, color=MUTED)

    xb2 = 4.3
    line([xz1, xb2], [YM, YM]); bus(xb2, "บัสจุดต่อ 22 kV", "nPcc  วัด SCR ที่นี่")

    yl = YM - 0.95
    line([xb2, xb2 + 0.75], [yl, yl]); line([xb2 + 0.75, xb2 + 0.75], [yl, 1.75])
    ax.add_patch(Rectangle((xb2 + 0.75 - 0.13, 2.35 - 0.13), 0.26, 0.26, fc=SURF, ec=INK, lw=1.6, zorder=3))
    ax.add_patch(Polygon([[xb2 + 0.57, 1.75], [xb2 + 0.93, 1.75], [xb2 + 0.75, 1.4]], closed=True, fc=INK, ec=INK))
    ax.text(xb2 + 1.05, 2.35, f"สวิตช์ปิดที่ t = {T_STEP:g} s", va="center", fontsize=9.5, color=ORANGE2)
    ax.text(xb2 + 0.75, 1.1, f"โหลด {LOAD / 1e3:.0f} kW", ha="center", va="top", fontsize=11, weight="bold")
    ax.text(xb2 + 0.75, 0.78, "ที่ระบบ 22 kV", ha="center", va="top", fontsize=9.5, color=INK2)

    yt = YM + 0.5
    xt1, xt2, rtf = 5.3, 5.64, 0.3
    line([xb2, xt1 - rtf], [yt, yt])
    ax.add_patch(Circle((xt1, yt), rtf, fill=False, ec=INK, lw=1.8))
    ax.add_patch(Circle((xt2, yt), rtf, fill=False, ec=INK, lw=1.8))
    ax.text((xt1 + xt2) / 2, yt + 0.5, "หม้อแปลง step up", ha="center", fontsize=10.5, weight="bold")
    ax.text((xt1 + xt2) / 2, yt - 0.45, f"{S / 1e6:g} MVA", ha="center", va="top", fontsize=9.5, color=INK2)
    ax.text((xt1 + xt2) / 2, yt - 0.75, "22 kV / 380 V", ha="center", va="top", fontsize=9.5, color=INK2)
    ax.text((xt1 + xt2) / 2, yt - 1.05, f"อิมพีแดนซ์ {ZT * 100:g}%", ha="center", va="top", fontsize=9.5, color=INK2)

    xb3 = 6.55
    line([xt2 + rtf, xb3], [yt, yt])
    ax.plot([xb3, xb3], [yt - 0.75, yt + 0.75], color=INK, lw=6, solid_capstyle="butt")
    ax.text(xb3, yt + 1.12, "บัส 380 V", ha="center", fontsize=11, weight="bold")
    ax.text(xb3, yt + 0.85, "nLV", ha="center", va="bottom", fontsize=9, color=MUTED)

    xi0 = 7.6
    line([xb3, xi0], [yt, yt])
    ax.add_patch(FancyBboxPatch((xi0, yt - 0.95), 3.1, 1.9, boxstyle="round,pad=0.02,rounding_size=0.12",
                                fc="#f3f7fd", ec=BLUE2, lw=1.6))
    sx, sy, sz = xi0 + 0.55, yt, 0.36
    ax.add_patch(Rectangle((sx - sz, sy - sz), 2 * sz, 2 * sz, fill=False, ec=INK, lw=1.4))
    ax.plot([sx - sz, sx + sz], [sy - sz, sy + sz], color=INK, lw=1.2)
    ax.plot([sx - sz * 0.75, sx - sz * 0.15], [sy + sz * 0.55, sy + sz * 0.55], color=INK, lw=1.2)
    ax.plot([sx - sz * 0.75, sx - sz * 0.15], [sy + sz * 0.38, sy + sz * 0.38], color=INK, lw=1.2, ls=(0, (2, 1.5)))
    u = np.linspace(0, 1, 40)
    ax.plot(sx + sz * 0.15 + sz * 0.6 * u, sy - sz * 0.5 + 0.06 * np.sin(u * 2 * np.pi), color=INK, lw=1.2)
    ax.text(xi0 + 1.05, yt + 0.62, "อินเวอร์เตอร์ GFL หรือ GFM", fontsize=10.5, weight="bold", color=BLUE2)
    ax.text(xi0 + 1.05, yt + 0.25, f"{S / 1e6:g} MVA จ่าย {P_SET / 1e3:.0f} kW ({P_SET / S:g} pu)", fontsize=9.5, color=INK2)
    ax.text(xi0 + 1.05, yt - 0.05, "ตัวกรอง LC ชุดเดียวกัน", fontsize=9.5, color=INK2)
    ax.text(xi0 + 1.05, yt - 0.35, "ตัวคุมตั้งค่าเป็น pu เท่ากัน", fontsize=9.5, color=INK2)
    ax.text(xi0 + 1.05, yt - 0.68, "SSN_GFL หรือ SSN_GFM", fontsize=8.5, color=MUTED)

    fig.text(0.02, 0.965, "ระบบที่จำลองใน DPsim", fontsize=14, weight="bold", va="top")
    fig.text(0.02, 0.915, "อินเวอร์เตอร์ 1 MVA ผ่านหม้อแปลง step up เข้าบัส 22 kV ซึ่งมีโหลดอยู่ด้วย ไล่ความแข็งของกริดต่าง ๆ "
             "จำลองแบบ EMT สามเฟส time step 50 µs", fontsize=10, color=INK2, va="top")
    return emit(fig, "fig1_system")


def fig2_map_gfl():
    """รูป 2 แผนที่เสถียรภาพของ GFL กับตัวอย่างขนาดการแกว่ง"""
    rows = map_gfl()
    vals, bnd = boundary_gfl_curve()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 5.6), gridspec_kw=dict(width_ratios=[1.15, 1]))
    fig.subplots_adjust(left=0.08, right=0.98, top=0.72, bottom=0.12, wspace=0.22)
    title(fig, "GFL ที่เร่ง PLL เร็ว แกว่งบนกริดอ่อน",
          "อินเวอร์เตอร์ 1 MVA ผ่านหม้อแปลง 6% เข้าบัส 22 kV วงควบคุมกระแส 500 Hz โครงข่ายจำลองใน DPsim แบบ EMT สามเฟส")
    scr_axis(a1)
    a1.set_ylabel("ความเร็ว PLL (เท่าของค่าตั้งต้น)"); a1.set_ylim(0, 14)
    ys = np.array([v for v, c in zip(vals, bnd) if c is not None])
    cs = np.array([c for c in bnd if c is not None])
    a1.fill_betweenx(ys, cs, 1.6, color=ORANGE2, alpha=0.10, lw=0)
    a1.plot(cs, ys, color=ORANGE2, lw=1.6, ls="--", zorder=2)
    done = set()
    for r in rows:
        kw = dict(STYLE[r["state"]])
        if r["state"] not in done:
            kw["label"] = r["state"]; done.add(r["state"])
        a1.scatter([r["scr"]], [r["pll"]], **kw, zorder=3)
    a1.plot([], [], color=ORANGE2, lw=1.6, ls="--", label="ขอบจากการไล่ SCR ละเอียด")
    a1.legend(loc="lower right", bbox_to_anchor=(1.0, 1.1), ncol=4, fontsize=9.5, handletextpad=0.3, columnspacing=0.9)
    a1.text(3.2, 13.1, "ไม่เสถียร", fontsize=11, color=ORANGE2)
    a1.text(900, 13.1, "เสถียร", fontsize=11, color=BLUE2)
    a1.text(0.0, 1.03, "แผนที่เสถียรภาพ", transform=a1.transAxes, fontsize=11, color=INK, va="bottom")

    # ขวา: ขนาดการแกว่งของสองจุดตัวอย่าง (PLL เร็ว 10 เท่า)
    for scr_, c_, txt in ((20, BLUE2, "SCR 20"), (4.2, ORANGE2, "SCR 4.2 (หลุด)")):
        r = pick(rows, scr=scr_, pll=10) or _point("gfl", scr_, pll=10)
        t, p, tb = cut_at_blowup(r["t"], r["p"])
        amp = swing(t, p, 0.02)
        k = (t > 0.02) & (t < 0.4)
        a2.plot(t[k], np.clip(amp[k], 1e-6, None), color=c_, lw=2.0, label=txt)
        if tb:
            a2.scatter([t[k][-1]], [min(amp[k][-1], 0.7)], marker="x", color=c_, s=70, lw=2.2, zorder=3)
    a2.set_yscale("log"); a2.set_ylim(1e-6, 1.0); a2.set_xlim(0.0, 0.4)
    a2.set_yticks([1e-6, 1e-4, 1e-2, 1]); a2.set_yticklabels(["0.000001", "0.0001", "0.01", "1"])
    a2.set_xlabel("เวลา (s)"); a2.set_ylabel("ขนาดการแกว่งของ P (pu)")
    a2.text(0.0, 1.03, "PLL เร็วขึ้น 10 เท่า ช่วง 0.4 วินาทีแรก", transform=a2.transAxes, fontsize=11, color=INK, va="bottom")
    a2.legend(loc="lower right", bbox_to_anchor=(1.0, 1.1), ncol=2, fontsize=10)
    return emit(fig, "fig2_map_gfl")


def fig3_map_gfm():
    """รูป 3 แผนที่เสถียรภาพของ GFM กับตัวอย่าง P ตามเวลา"""
    rows = map_gfm()
    xs = np.array(list(np.linspace(0, 0.03, 16)))
    b = boundary_gfm(list(xs))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 5.6), gridspec_kw=dict(width_ratios=[1.15, 1]))
    fig.subplots_adjust(left=0.08, right=0.98, top=0.72, bottom=0.12, wspace=0.22)
    title(fig, "GFM บนกริดแข็งแกว่งได้ และ virtual reactance ช่วยได้",
          f"อินเวอร์เตอร์ 1 MVA ผ่านหม้อแปลง 6% เข้าบัส 22 kV ต่อโหลด {LOAD / 1e3:.0f} kW ที่ 1 วินาที ผลจาก DPsim แบบ EMT สามเฟส")
    scr_axis(a1)
    a1.set_ylabel("virtual reactance (pu)"); a1.set_ylim(-0.003, 0.034)
    xx = np.array([x for x, s in zip(xs, b) if s is not None]); ss = np.array([s for s in b if s is not None])
    a1.fill_betweenx(xx, 1300, ss, color=ORANGE2, alpha=0.10, lw=0)
    a1.plot(ss, xx, color=ORANGE2, lw=1.6, ls="--", zorder=2)
    done = set()
    for r in rows:
        kw = dict(STYLE[r["state"]])
        if r["state"] not in done:
            kw["label"] = r["state"]; done.add(r["state"])
        a1.scatter([r["scr"]], [r["xv"]], **kw, zorder=3)
    a1.plot([], [], color=ORANGE2, lw=1.6, ls="--", label="ขอบจาก eigenvalue")
    a1.legend(loc="lower right", bbox_to_anchor=(1.0, 1.1), ncol=4, fontsize=9.5, handletextpad=0.3, columnspacing=0.9)
    a1.text(900, 0.0305, "ไม่เสถียร", fontsize=11, color=ORANGE2)
    a1.text(5, 0.0305, "เสถียร", fontsize=11, color=BLUE2)
    a1.text(0.0, 1.03, "แผนที่เสถียรภาพ", transform=a1.transAxes, fontsize=11, color=INK, va="bottom")

    for scr_, c_, txt in ((20, BLUE2, "SCR 20"), (50, ORANGE2, "SCR 50 แกว่งโตขึ้น")):
        r = pick(rows, scr=scr_, xv=0.0) or _point("gfm", scr_, xv=0.0)
        t, p, tb = cut_at_blowup(r["t"], r["p"])
        k = t > 0.0
        a2.plot(t[k], p[k], color=c_, lw=1.3, label=txt)
        if tb:
            a2.scatter([t[k][-1]], [np.clip(p[k][-1], 0.25, 1.45)], marker="x", color=c_, s=70, lw=2.2, zorder=3)
    a2.set_xlim(0.0, 3.0); a2.set_ylim(0.2, 1.5)
    a2.set_xlabel("เวลา (s)"); a2.set_ylabel("P ที่อินเวอร์เตอร์จ่าย (pu)")
    a2.axvline(1.0, color=MUTED, lw=1.0, ls=":")
    a2.text(1.04, 1.47, f"ต่อโหลด {LOAD / 1e3:.0f} kW", fontsize=9, color=MUTED, va="top")
    a2.text(0.0, 1.03, "ไม่มี virtual reactance", transform=a2.transAxes, fontsize=11, color=INK, va="bottom")
    a2.legend(loc="lower right", bbox_to_anchor=(1.0, 1.1), ncol=2, fontsize=10)
    return emit(fig, "fig3_map_gfm")


def fig4_damping():
    """รูป 4 damping ratio ของโหมดที่หน่วงน้อยที่สุด ทับด้วยจุดผลจาก DPsim"""
    z = zeta_curves(); scr = z["scr"]
    rows_gfl, rows_gfm = map_gfl(), map_gfm()
    fig, axs = plt.subplots(1, 2, figsize=(11, 5.4), sharey=True)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.74, bottom=0.13, wspace=0.08)
    title(fig, "damping ratio ของโหมดที่หน่วงน้อยที่สุด เมื่อไล่ความแข็งของกริด",
          "เส้นคำนวณจาก eigenvalue (โหมดต่ำกว่า 400 Hz) ต่ำกว่าศูนย์คือแกว่งขยาย  จุดคือผลที่รันใน DPsim  วงกลมนิ่ง กากบาทไม่นิ่ง")
    series = (
        (axs[0], rows_gfl, "pll", "GFL", BLUE2,
         [(1, BLUE4, "PLL ค่าตั้งต้น"), (4, BLUE3, "PLL เร็วขึ้น 4 เท่า"),
          (8, BLUE2, "PLL เร็วขึ้น 8 เท่า"), (12, BLUE, "PLL เร็วขึ้น 12 เท่า")]),
        (axs[1], rows_gfm, "xv", "GFM", ORANGE2,
         [(0.026, ORANGE3, "virtual reactance 0.026 pu"), (0.01, ORANGE2, "virtual reactance 0.01 pu"),
          (0.0, ORANGE, "ไม่มี virtual reactance")]),
    )
    for ax, rows, key, lab, col, items in series:
        for v, c_, label in items:
            y = z[f"{'gfl' if key == 'pll' else 'gfm'}{v}"] * 100
            ax.plot(scr, y, color=c_, lw=2.2, label=label)
            for r in rows:
                if abs(r[key] - v) > 1e-9 or r["scr"] < scr[0]:
                    continue
                yy = float(np.clip(np.interp(r["scr"], scr, y), -38, 100))
                if r["state"] == "นิ่ง":
                    ax.scatter([r["scr"]], [yy], marker="o", s=26, color=c_, edgecolor=SURF, lw=0.8, zorder=3)
                else:
                    ax.scatter([r["scr"]], [yy], marker="x", s=55, color=c_, lw=2.0, zorder=3)
        ax.axhline(0, color=INK, lw=1.0)
        ax.set_xscale("log"); ax.set_xlim(1.75, 1000)
        ax.set_xticks([2, 5, 10, 20, 50, 100, 300, 1000]); ax.set_xticklabels(["2", "5", "10", "20", "50", "100", "300", "1000"])
        ax.minorticks_off(); ax.set_ylim(-40, 105)
        ax.set_xlabel("SCR ที่บัส 22 kV (สเกล log)")
        ax.text(0.0, 1.03, lab, transform=ax.transAxes, fontsize=13, weight="bold", color=col, va="bottom")
        ax.legend(loc="center right" if lab == "GFL" else "upper right",
                  bbox_to_anchor=(1.0, 0.46) if lab == "GFL" else None, fontsize=9.5)
    axs[0].set_ylabel("damping ratio (%)")
    return emit(fig, "fig4_damping")


def fig5_phase_jump():
    """รูป 5 มุมเฟสของกริดกระโดดที่ SCR 2.5 ช่วง 150 ms แรก"""
    fig, axs = plt.subplots(2, 1, figsize=(10, 7.0), sharex=True)
    fig.subplots_adjust(left=0.1, right=0.97, top=0.8, bottom=0.09, hspace=0.12)
    title(fig, f"มุมเฟสของกริดกระโดด {abs(JUMP_DEG):g} องศา ที่ SCR 2.5",
          "ช่วง 150 มิลลิวินาทีแรก ใช้ค่าตั้งต้นทั้งคู่ กระแสเป็น pu ของพิกัดอินเวอร์เตอร์ ผลจาก DPsim")
    for kind, label, c_ in (("gfl", "GFL", BLUE2), ("gfm", "GFM", ORANGE2)):
        d = jump(kind, 2.5)
        t = (d["t"] - 1.0) * 1e3; k = (t > -10) & (t < 150)
        axs[0].plot(t[k], d["p"][k], color=c_, lw=1.8, label=label)
        axs[1].plot(t[k], d["i"][k], color=c_, lw=1.8, label=label)
    for ax in axs:
        ax.axvline(0, color=MUTED, lw=1.0, ls=":")
    axs[0].set_ylabel("P (pu)"); axs[1].set_ylabel("กระแส (pu)")
    axs[1].set_xlabel("เวลาหลังมุมกระโดด (ms)")
    axs[0].legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=2, fontsize=10.5)
    return emit(fig, "fig5_phase_jump")


def fig6_fast_pll():
    """รูป 6 มุมกระโดดเดียวกันที่ SCR 5 ดูความถี่ที่ตัวควบคุมใช้"""
    fig, ax = plt.subplots(figsize=(10, 5.2))
    fig.subplots_adjust(left=0.1, right=0.97, top=0.74, bottom=0.12)
    title(fig, "มุมกระโดดเท่าเดิม ถ้าเร่ง PLL ให้ GFL ตามทันเร็วขึ้น",
          f"SCR 5 มุมกริดกระโดด {abs(JUMP_DEG):g} องศา แกนตั้งคือความถี่ที่ PLL ของ GFL หรือตัวควบคุมของ GFM ใช้อยู่ ผลจาก DPsim")
    for kind, pll_, label, c_, lw in (("gfl", 1, "GFL PLL ค่าตั้งต้น", BLUE3, 1.8),
                                      ("gfl", 8, "GFL PLL เร็วขึ้น 8 เท่า", BLUE, 1.2),
                                      ("gfm", 1, "GFM", ORANGE2, 2.2)):
        d = jump(kind, 5, pll=pll_)
        t = d["t"] - 1.0; k = (t > -0.02) & (t < 0.2)
        ax.plot(t[k], d["f"][k], color=c_, lw=lw, label=label)
    ax.axvline(0, color=MUTED, lw=1.0, ls=":")
    ax.axhspan(37, 47.5, color=ORANGE2, alpha=0.06, lw=0)
    ax.text(0.195, 39.0, "ต่ำกว่า 47.5 Hz", fontsize=9, color=MUTED, ha="right")
    ax.set_ylim(38, 60); ax.set_xlim(-0.02, 0.2)
    ax.set_ylabel("ความถี่ (Hz)"); ax.set_xlabel("เวลาหลังมุมกระโดด (s)")
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=3, fontsize=10)
    # ความถี่ของ GFM ตกแค่ราว 0.08 Hz จึงขยายแกนตั้งในกรอบเล็ก
    d = jump("gfm", 5); t = d["t"] - 1.0
    ins = ax.inset_axes([0.47, 0.68, 0.5, 0.25])
    k = (t >= 0.0) & (t <= 0.3)
    ins.plot((t[k]) * 1e3, d["f"][k], color=ORANGE2, lw=1.2)
    ins.set_facecolor(SURF); ins.grid(False); ins.tick_params(labelsize=8, colors=MUTED)
    for sp in ins.spines.values():
        sp.set_color(AXIS)
    ins.set_ylim(49.9, 50.01); ins.set_yticks([49.92, 49.96, 50.0])
    ins.set_xlabel("ms", fontsize=8, color=MUTED, labelpad=1)
    ins.set_title("GFM ขยายแกนตั้ง ช่วง 300 ms แรก", fontsize=8.5, color=INK2, pad=3)
    return emit(fig, "fig6_fast_pll")


def fig7_pros_cons():
    """รูป 7 ข้อดีและข้อจำกัด (เรียบเรียงเอง อ้างอิง NREL 2020 และ IEEE TPWRS 2022)"""
    COLS = [
        ("GFL", "ทำตัวเป็นแหล่งจ่ายกระแส ตามมุมแรงดันที่ PLL วัดได้", BLUE2,
         ["ตัวควบคุมไม่ซับซ้อน และใช้กันแพร่หลายอยู่แล้ว",
          "สั่งกำลังได้เร็ว คุมแรงดัน DC ของโซลาร์และลมได้ดี",
          "จำกัดกระแสตอนลัดวงจรได้ในตัว เพราะคุมกระแสอยู่แล้ว"],
         ["ต้องมีกริดแข็งพอให้ PLL จับมุมแรงดันได้",
          "มุมกริดกระโดด ต้องรอ PLL ล็อกมุมใหม่|ความถี่ที่วัดได้ส่ายไปชั่วขณะ",
          "เร่ง PLL ให้ไวบนกริดอ่อน อาจแกว่งจนไม่กลับ",
          "ไม่ช่วยให้กริดแข็งขึ้น สร้างแรงดันเอง|และ black start ไม่ได้"]),
        ("GFM", "ทำตัวเป็นแหล่งจ่ายแรงดัน สร้างมุมและขนาดแรงดันเอง", ORANGE2,
         ["สร้างแรงดันและความถี่เองได้|ทำงานแยกเกาะและ black start ได้",
          "ยิ่งกริดอ่อน การแกว่งยิ่งหน่วงได้ดี",
          "มุมกริดเปลี่ยน กำลังตอบสนองทันที|ไม่ต้องรอวัดก่อน",
          "ใช้ได้แม้ระบบมีแต่อินเวอร์เตอร์"],
         ["บนกริดแข็งมากอาจแกว่ง ต้องเพิ่ม|virtual reactance ช่วยหน่วง",
          "กระแสพุ่งตามทันทีเมื่อกริดเปลี่ยน ยิ่งกริดแข็ง|ยิ่งพุ่งสูง ต้องมีวิธีจำกัดกระแสที่ดี",
          "มาตรฐานและประสบการณ์ใช้งานจริงยังน้อย"]),
    ]
    reg = dict(fontproperties=FONT_REG) if FONT_REG else {}
    bold = dict(fontproperties=FONT_BOLD) if FONT_BOLD else dict(weight="bold")
    fig = plt.figure(figsize=(10.0, 9.6), dpi=150, facecolor=SURF)
    LINE = 0.024

    def rule(x0, x1, y, color=RULE, lw=0.9):
        fig.add_artist(plt.Line2D([x0, x1], [y, y], color=color, lw=lw))

    fig.text(0.02, 0.975, "ข้อดีและข้อจำกัดของ GFL กับ GFM", fontsize=14, color=INK, va="top", **bold)
    lowest = 1.0
    for (name, desc, color, pros, cons), x in zip(COLS, [0.02, 0.52]):
        y = 0.895; CW = 0.46
        fig.text(x, y, name, fontsize=17, color=color, va="center", **bold)
        fig.text(x, y - 0.034, desc, fontsize=10.5, color=INK2, va="center", **reg)
        y -= 0.062
        rule(x, x + CW, y, color=color, lw=1.6)
        for label, items in (("ข้อดี", pros), ("ข้อจำกัด", cons)):
            y -= 0.036
            fig.text(x, y, label, fontsize=12, color=INK, va="center", **bold)
            y -= 0.02
            for text_ in items:
                lines = text_.split("|")
                y -= 0.022
                for i, ln in enumerate(lines):
                    fig.text(x, y - i * LINE, ln, fontsize=11.5, color=INK, va="center", **reg)
                y -= (len(lines) - 1) * LINE + 0.026
                rule(x, x + CW, y)
        lowest = min(lowest, y)
    fig.text(0.02, lowest - 0.035,
             "อ้างอิง  Lin et al., Research Roadmap on Grid-Forming Inverters, NREL 2020  และ  "
             "Li, Gu, Green, IEEE Trans. Power Systems 2022", fontsize=9, color=MUTED, va="center", **reg)
    return emit(fig, "fig7_pros_cons", crop_bottom=(1 - (lowest - 0.035)) + 0.03)


if __name__ == "__main__":
    import sys
    t0 = time.time()
    if len(sys.argv) > 1:
        OUTDIR = Path(sys.argv[1])
    fig1_system()
    fig2_map_gfl()
    fig3_map_gfm()
    fig4_damping()
    fig5_phase_jump()
    fig6_fast_pll()
    fig7_pros_cons()
    print(f"เสร็จใน {time.time() - t0:.0f} วินาที")
