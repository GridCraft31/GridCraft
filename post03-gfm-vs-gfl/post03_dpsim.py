# -*- coding: utf-8 -*-
"""GFL กับ GFM เมื่อความแข็งของกริดเปลี่ยนไป จำลองด้วย DPsim (EMT สามเฟส)

ระบบ: แหล่งจ่ายกริด 22 kV -> อิมพีแดนซ์กริด (ตาม SCR) -> บัส 22 kV (มีโหลด 50 kW รอต่อ)
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
QUICK = True            # True รันเฉพาะแถวตัวอย่างของแผนที่ (ราว 1 นาที), False รันครบทุกจุด (หลายนาที)
SCR_LIST = [1000, 300, 100, 50, 20, 10, 5, 3, 2.5, 2, 1.8]
PLL_LIST = [1, 2, 4, 6, 8, 10, 12]          # ความเร็ว PLL ของ GFL เป็นเท่าของค่าตั้งต้น
XV_LIST = [0.0, 0.01, 0.026]          # virtual reactance ของ GFM (pu)
JUMP_DEG = -5.0                       # มุมเฟสกริดกระโดด (องศา)

F = 50.0; W = 2 * np.pi * F
DT, T_END, T_STEP = 50e-6, 3.0, 1.0   # time step, เวลาจำลอง, เวลาที่ต่อโหลด
S = 1e6                               # พิกัดอินเวอร์เตอร์ 1 MVA
VLV = np.sqrt(3) * 220.0              # 380 V
VPK = np.sqrt(2) * 220.0
XR, ZT, XRT = 10.0, 0.06, 8.0         # X/R กริด, อิมพีแดนซ์หม้อแปลง (pu), X/R หม้อแปลง
P_SET = 0.8 * S
LOAD = 0.05 * S                       # โหลดที่ต่อที่ T_STEP

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
# 3) รันและวาดรูป
# =====================================================================
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


def damping_plot(pll_list=(1, 4, 8, 12), xv_list=(0.026, 0.01, 0.0)):
    scr = np.logspace(np.log10(1.75), 3, 60)
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    blues = ["#a9c9f0", "#6fa6e6", "#2a78d6", "#1f5fae"]; oranges = ["#f3a37f", "#eb6834", "#c4501f"]
    for pll, c_ in zip(pll_list, blues):
        axs[0].plot(scr, [100 * damping_ratio(eigen_gfl(s, pll)) for s in scr], color=c_, lw=2, label=f"PLL x{pll}")
    for xv, c_ in zip(xv_list, oranges):
        axs[1].plot(scr, [100 * damping_ratio(eigen_gfm(s, xv)) for s in scr], color=c_, lw=2, label=f"XV {xv} pu")
    for ax, lab in zip(axs, ("GFL", "GFM")):
        ax.axhline(0, color="k", lw=1); ax.set_xscale("log"); ax.set_ylim(-40, 105); ax.grid(alpha=0.3)
        ax.set_title(lab); ax.set_xlabel("SCR ที่บัส 22 kV"); ax.legend(fontsize=8)
    axs[0].set_ylabel("damping ratio (%)")
    plt.show()


def swing(t, y, win):
    """ขนาดการแกว่งยอดถึงยอดแบบเลื่อนหน้าต่าง ใช้กับ GFL ที่แกว่งเร็วจนเส้นทึบ"""
    n = max(3, int(round(win / (t[1] - t[0])))); ker = np.ones(n) / n
    avg = np.convolve(np.pad(y, (n - 1, 0), mode="edge"), ker, mode="valid")
    return 2 * np.sqrt(2) * np.sqrt(np.convolve(np.pad((y - avg) ** 2, (n - 1, 0), mode="edge"), ker, mode="valid"))


COLOR = {"นิ่ง": "#2a78d6", "แกว่งไม่หาย": "#898781", "หลุด": "#eb6834"}


def stability_map(kind):
    rows = PLL_LIST if kind == "gfl" else XV_LIST
    if QUICK:
        rows = [10] if kind == "gfl" else [0.0]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for v in rows:
        for scr in SCR_LIST:
            r = run_step(kind, scr, pll=v) if kind == "gfl" else run_step(kind, scr, xv=v)
            ax.scatter(scr, v, color=COLOR[r["state"]], s=60, zorder=3)
            print(f"{kind} {v} SCR {scr}: {r['state']}")
    if kind == "gfl":
        vals = list(GFL_BOUNDARY_POST)
        b = list(GFL_BOUNDARY_POST.values()) if QUICK else boundary_gfl(vals)
        lab = "ขอบจากการไล่ SCR ใน DPsim"
    else:
        vals = list(np.linspace(0, 0.03, 16)); b = boundary_gfm(vals); lab = "ขอบจาก eigenvalue"
    ax.plot([x for x in b if x], [v for v, x in zip(vals, b) if x], "--", color="#eb6834", label=lab)
    for k_, c_ in COLOR.items():
        ax.scatter([], [], color=c_, label=k_)
    ax.set_xscale("log"); ax.invert_xaxis(); ax.set_xlabel("SCR ที่บัส 22 kV (ยิ่งไปทางขวากริดยิ่งอ่อน)")
    ax.set_ylabel("ความเร็ว PLL (เท่า)" if kind == "gfl" else "virtual reactance (pu)")
    ax.set_title("GFL" if kind == "gfl" else "GFM"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    plt.show()


def phase_jump_plots():
    gfl = run_jump("gfl", 2.5); gfm = run_jump("gfm", 2.5)
    fig, axs = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    for r, lab, c in ((gfl, "GFL", "#2a78d6"), (gfm, "GFM", "#eb6834")):
        t = (r["t"] - 1.0) * 1e3; k = (t > -10) & (t < 150)
        axs[0].plot(t[k], r["p"][k], color=c, label=lab); axs[1].plot(t[k], r["i"][k], color=c, label=lab)
    axs[0].set_ylabel("P (pu)"); axs[1].set_ylabel("กระแส (pu)"); axs[1].set_xlabel("เวลาหลังมุมกระโดด (ms)")
    axs[0].set_title(f"มุมกริดกระโดด {abs(JUMP_DEG):g} องศา ที่ SCR 2.5"); axs[0].legend(); plt.show()

    fig, ax = plt.subplots(figsize=(8, 4))
    for kind, pll, lab, c in (("gfl", 1, "GFL PLL ค่าตั้งต้น", "#6fa6e6"), ("gfl", 8, "GFL PLL เร็ว 8 เท่า", "#1f5fae"),
                              ("gfm", 1, "GFM", "#eb6834")):
        r = run_jump(kind, 5, pll=pll); t = r["t"] - 1.0; k = (t > -0.02) & (t < 0.2)
        ax.plot(t[k], r["f"][k], color=c, label=lab)
    ax.set_ylim(38, 60); ax.set_xlabel("เวลาหลังมุมกระโดด (s)"); ax.set_ylabel("ความถี่ที่ตัวควบคุมใช้ (Hz)")
    # ความถี่ของ GFM ตกแค่หลักร้อยของ Hz จึงขยายแกนตั้งให้ดูในกรอบเล็ก
    ins = ax.inset_axes([0.47, 0.62, 0.5, 0.3]); k = (t >= 0) & (t <= 0.3)
    ins.plot(t[k] * 1e3, r["f"][k], color="#eb6834"); ins.set_title("GFM ขยายแกนตั้ง", fontsize=9)
    ins.set_xlabel("ms", fontsize=8); ins.tick_params(labelsize=8)
    ax.set_title("มุมกระโดดที่ SCR 5"); ax.legend(loc="lower right"); plt.show()


def thai_font():
    """ใช้ฟอนต์ไทยถ้ามี (บน Colab ติดตั้งด้วย !apt-get install -y fonts-thai-tlwg)"""
    from matplotlib import font_manager
    path = Path("/usr/share/fonts/truetype/tlwg/Garuda.ttf")
    if path.exists():
        font_manager.fontManager.addfont(str(path))
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(path)).get_name()
    plt.rcParams["axes.unicode_minus"] = False


thai_font()

if __name__ == "__main__":
    t0 = time.time()
    stability_map("gfl")
    stability_map("gfm")
    damping_plot()
    phase_jump_plots()
    print(f"เสร็จใน {time.time() - t0:.0f} วินาที")
