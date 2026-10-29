# TimingPredict — complete DNA analysis of `C:\Users\supre\Downloads\TimingPredict`

*Written 2026-10-18. Every number in this document was verified against the files
on disk in this folder: file reads, record counts, a `build_graph.py` run over 7
designs, and a DGL load of `spm.graph.bin`. Nothing is quoted from the paper that
is not also in the code or README.*

---

## 1. What this folder actually is

Three distinct things are stacked in one directory, and almost all confusion
about this repo comes from not separating them:

| Layer | What it is | Where it lives | Size |
|---|---|---|---|
| **A — the published model** | The official code for the DAC 2022 paper *"A Timing Engine Inspired Graph Neural Network Model for Pre-Routing Slack Prediction"* (Guo, Liu, Gu, Zhang, Pan, Lin) | `model.py`, `data_graph.py`, `train_gnn.py`, `tee.py` + 6 baseline files | ~60 KB |
| **B — the raw dataset** | 21 open-source chip designs taken all the way through place-and-route in SkyWater sky130, captured as the actual EDA files (Verilog, DEF, SPEF, SDC, SDF) | `netlists/` + `netlists/techlib/` | **3.5 GB** |
| **C — a locally built ingest pipeline** | A second, independent implementation that *rebuilds* the model's binary graphs from layer B, plus an OpenLane path to add brand-new designs (a GPU) to the dataset | `build_graph.py`, `liberty_lut.py`, `sta_ref.py`, `make_dataset.py`, `device.py`, `openlane/` | ~70 KB |

Layer C was added locally on 13–16 Oct 2025 by whoever set this folder up. It is
not part of the upstream repo; `git status` shows those files as untracked.

### The problem being solved

In a chip design flow, you only learn the *real* timing of a circuit after
**detailed routing** — because wire delay does not exist until wires exist.
Routing a block takes hours. So designers spend those hours, look at the timing
report, discover a violation, change the placement, and route again.

The task here is: **predict post-route timing from pre-route information.** Given
where the cells were placed, how are they going to be connected and how slow will
the result be? Specifically the model predicts four things per design:

1. **net delay** — the wire delay of each driver→sink connection
2. **cell delay** — the delay through each logic gate
3. **arrival time (AT)** — when a signal actually reaches each pin
4. **slack** — how much timing margin is left (the paper's headline metric)

### The data flow in one line

```
RTL ──OpenLane──> placed+routed design ──OpenSTA──> SDF with real delays
                       │                                 │
                       └──> pin coordinates ─┐           └──> delays + arrival times
                                             ▼
                              DGL heterograph (pins = nodes, timing arcs = edges)
                                             │
                                             ▼
                              TimingGCN  ──>  delay / AT / slack prediction
```

---

## 2. The technology — "is this 130nm, 90nm, …?"

**One process node, one standard-cell library, two PVT corners. No mixing.**

### Process node: SkyWater sky130 — 130 nm CMOS

This is the open-source SkyWater 130 nm PDK (the one used by the Google/efabless
free shuttle program). Evidence on disk:

- Every single cell master in every netlist is named `sky130_fd_sc_hd__*`
- `netlists/techlib/merged_unpadded.lef` carries the SkyWater PDK Apache-2.0
  copyright header, LEF `VERSION 5.7`, and **441 `MACRO` definitions**
- `MANUFACTURINGGRID 0.005 ;` (5 nm grid), `DATABASE MICRONS 1000`
- The placement site is

  ```
  SITE unithd
    SYMMETRY Y ;
    CLASS CORE ;
    SIZE 0.46 BY 2.72 ;
  END unithd
  ```

  — a 0.46 µm × 2.72 µm standard-cell row, which is the sky130 high-density site.
  You can see it in the DEFs: `ROW ROW_0 unithd 5520 10880 N DO 177 BY 1 STEP 460 0`
  (460 database units = 0.46 µm pitch, rows 2720 units = 2.72 µm apart).

### Standard-cell library: `sky130_fd_sc_hd`

"fd" = foundry digital, "sc" = standard cell, **"hd" = high density**. This is
the default OpenLane library for sky130. `picorv32a` alone instantiates **132
distinct cell masters** out of the 441 available.

A real cell mix, from `picorv32a.synthesis_preroute.v` (top entries):

| count | master | kind |
|---|---|---|
| 8,381 | `decap_12` | physical only (decoupling cap) |
| 7,787 | `decap_4` | physical only |
| 6,000 | `tapvpwrvgnd_1` | physical only (well tap) |
| 5,899 | `fill_1` | physical only (filler) |
| 1,855 | `a2bb2o_1` | logic (AND-OR) |
| 1,570 | `inv_2` | logic (inverter) |
| 1,273 | `o22a_1` | logic (OR-AND) |
| 1,239 | `dfxtp_1` | sequential (D flip-flop) |
| 1,088 | `clkbuf_2` | clock buffer |
| 1,030 | `mux2_1` | logic (2:1 mux) |
| 919 | `diode_2` | antenna diode |

Note how dominant the physical-only cells are — that matters in §3.

### Operating corners: exactly two, and they define the whole tensor layout

`netlists/techlib/` ships two Liberty files, and *nothing else*:

| Tag in code | Liberty file | Size | Voltage | Temp | STA role |
|---|---|---|---|---|---|
| **`E`** (early / fast) | `sky130_fd_sc_hd__ff_n40C_1v95.lib` | 71.3 MB | 1.95 V | −40 °C | `read_liberty -min` |
| **`L`** (late / slow) | `sky130_fd_sc_hd__ss_100C_1v60.lib` | 12.8 MB | 1.60 V | 100 °C | `read_liberty -max` |

`ff` = fast-fast silicon, `ss` = slow-slow. From the `ss` file's header:

```
technology("cmos");
delay_model : "table_lookup";
time_unit : "1ns";  capacitive_load_unit(1.0, "pf");  voltage_unit : "1V";
operating_conditions ("ss_100C_1v60") {
    voltage : 1.6;  process : 1.0;  temperature : 100.0;
    tree_type : "balanced_tree";
}
default_max_transition : 1.5;
```

**This two-corner choice is the single most load-bearing fact about the data
format.** Every timing quantity in the entire project — node features, edge
features, labels — is a **4-vector ordered `[E_rise, E_fall, L_rise, L_fall]`**.
Two corners × two signal edges = 4. If you remember only one thing about the
tensor layout, remember that.

### The delay-table shape, and why the model's feature is 7×7

sky130's delay tables all use one template:

```
lu_table_template ("del_1_7_7") {
    variable_1 : "input_net_transition";
    variable_2 : "total_output_net_capacitance";
    index_1("1, 2, 3, 4, 5, 6, 7");
    index_2("1, 2, 3, 4, 5, 6, 7");
}
```

7 input-slew points × 7 output-load points. That is *natively* 7×7, so the model
can consume the library tables with **no resampling at all** — which is exactly
why `liberty_lut.py` sets `LUT_SZ = 7` and the cell edge feature is 512 wide
(§5). A different PDK with, say, 8×8 tables would need interpolation.

### Per-design targets

Clock periods were set per design by whoever hardened them — they are **not**
uniform, and they range over 6.5×:

| fastest | | slowest | |
|---|---|---|---|
| `cic_decimator`, `genericfir`, `spm` | 10.00 ns | `blabla` | 65.00 ns |
| `aes_cipher` | 10.09 ns | `synth_ram` | 50.00 ns |
| `aes128` | 11.43 ns | `salsa20` | 36.73 ns |

Die areas span 93 × 103 µm (`spm`) to 1684 × 1695 µm (`jpeg_encoder`) — a factor
of ~300 in area.

---

## 3. The dataset — 21 designs

This is the content of the companion **TimingPredict/Dataset** repo. Each of the
21 entries is an open-source RTL block taken through the same flow:

```
RTL ─> Yosys (synthesis) ─> OpenROAD (floorplan, place, CTS, route)
    ─> OpenRCX (parasitic extraction) ─> OpenSTA (timing) ─> files captured here
```

All 21 are **post-route** snapshots. That is essential: the whole point is to
have ground truth that only exists after routing.

### The full table

Columns: `clk` from the SDC `create_clock`; `comps` from the DEF `COMPONENTS`
header; `die` from `DIEAREA`; and the four graph counts, which are the published
reference numbers hardcoded in `build_graph.py:REF` and which the local builder
reproduces exactly (§8).

| design | clk (ns) | DEF comps | die (µm) | graph nodes | net edges | cell edges | endpoints | what it is |
|---|---|---|---|---|---|---|---|---|
| `spm` | 10.00 | 1,028 | 93 × 103 | 1,121 | 765 | 700 | 129 | serial-parallel multiplier |
| `cic_decimator` | 10.00 | 2,676 | 142 × 153 | 3,131 | 2,232 | 2,102 | 130 | CIC decimation filter |
| `usb` | 12.55 | 3,376 | 169 × 180 | 3,361 | 2,406 | 2,189 | 344 | USB core |
| `zipdiv` | 19.09 | 4,022 | 176 × 186 | 4,398 | 3,102 | 2,913 | 181 | divider (ZipCPU) |
| `usb_cdc_core` | 15.60 | 6,729 | 223 × 234 | 7,406 | 5,200 | 4,869 | 630 | USB CDC |
| `xtea` | 26.03 | 8,136 | 236 × 247 | 10,213 | 7,151 | 6,882 | 423 | XTEA block cipher |
| `wbqspiflash` | 18.86 | 9,017 | 262 × 272 | 9,672 | 6,798 | 6,454 | 323 | Wishbone QSPI flash ctrl |
| `synth_ram` | 50.00 | 24,305 | 458 × 469 | 25,910 | 19,024 | 16,782 | 2,112 | flop-based RAM |
| `BM64` | 20.00 | 57,983 | 711 × 721 | 38,458 | 27,843 | 25,334 | 1,800 | Bluetooth/audio block |
| `genericfir` | 10.00 | 34,719 | 549 × 559 | 38,827 | 28,845 | 25,013 | 3,811 | FIR filter |
| `y_huff` | 11.35 | 42,451 | 565 × 576 | 48,216 | 33,689 | 30,612 | 2,391 | Huffman coder (JPEG) |
| `blabla` | 65.00 | 123,336 | 1056 × 1067 | 55,568 | 39,853 | 35,689 | 1,614 | RISC-V-ish core |
| `picorv32a` | 24.73 | 57,366 | 660 × 670 | 58,676 | 43,047 | 40,208 | 1,920 | PicoRV32 RISC-V CPU |
| `aes_cipher` | 10.09 | 65,607 | 703 × 713 | 59,777 | 42,671 | 41,411 | 660 | AES cipher core |
| `des` | 20.00 | 59,687 | 683 × 694 | 60,541 | 44,478 | 41,845 | 2,048 | DES cipher |
| `usbf_device` | 14.60 | 59,757 | 652 × 663 | 66,345 | 46,241 | 42,226 | 4,404 | USB function device |
| `salsa20` | 36.73 | 121,934 | 1040 × 1051 | 78,486 | 57,737 | 52,895 | 3,710 | Salsa20 stream cipher |
| `aes128` | 11.43 | 238,867 | 1355 × 1365 | 211,045 | 148,997 | 138,457 | 5,696 | AES-128 |
| `aes192` | 12.37 | 327,794 | 1647 × 1658 | 234,211 | 165,350 | 152,910 | 8,096 | AES-192 |
| `jpeg_encoder` | 15.78 | 337,006 | 1684 × 1695 | 238,216 | 176,737 | 167,960 | 4,422 | JPEG encoder |
| `aes256` | 15.00 | 309,677 | 1533 × 1544 | 290,955 | 207,414 | 189,262 | 11,200 | AES-256 |

Range: **1,121 → 290,955 nodes**, a factor of 260. That spread is deliberate —
it is what lets the paper claim the model generalises across design sizes.

### Two things this table reveals

**(a) `DEF COMPONENTS` is much larger than `cell edges`.** `picorv32a` places
57,366 components but yields only 40,208 cell timing arcs. That is not an error:
roughly 50–70 % of placed instances are **physical-only** cells —
`decap_*` (decoupling capacitors), `fill_*` (filler), `tapvpwrvgnd_*` (well
taps). They have no signal pins and no timing arcs. `build_graph.py` drops them
with

```python
PHYS_RE = re.compile(r'__(fill|decap|tap|tapvpwrvgnd)')  # diode has a real signal pin
```

Note the comment: antenna `diode_*` cells are *kept*, because their `DIODE` pin is
a genuine timing sink that loads the net.

**(b) The design mix is narrow.** Counting by function:

- **8 of 21 are crypto datapaths**: `aes128`, `aes192`, `aes256`, `aes_cipher`,
  `des`, `salsa20`, `xtea` (+ `spm`, an arithmetic datapath)
- **4 are USB / serial interfaces**: `usb`, `usb_cdc_core`, `usbf_device`,
  `wbqspiflash`
- **3 are DSP**: `cic_decimator`, `genericfir`, `y_huff`
- **2–3 are CPUs**: `picorv32a`, `blabla`, `zipdiv`
- **1 is memory**: `synth_ram`
- **no GPU, no AI accelerator, no large control-dominated SoC**

The AES family alone is 4 entries of the same architecture at three key widths,
so the *effective* diversity is lower than 21. This narrowness is exactly the
motivation for layer C's `openlane/` directory (§6c), which exists to add a GPU
as a 22nd design.

---

## 4. The seven raw file types, and exactly how they connect

Every design directory holds the same seven files. Using `netlists/spm/` (the
smallest, 2.2 MB total) as the worked example:

```
netlists/spm/
├── spm.synthesis_preroute.v   129 KB  gate-level netlist      --> NODES
├── 20-spm.def                 310 KB  routed placement        --> COORDINATES
├── 20-spm.spef                257 KB  extracted RC parasitics --> (feeds OpenSTA)
├── spm.cts_1.sdc              8.6 KB  timing constraints      --> (feeds OpenSTA)
├── spm.opensta.tcl            440  B  the recipe              --> explains everything
├── spm.sdf                    322 KB  delays                  --> DELAY LABELS
└── spm.at.sdf                1.23 MB  delays + arrival times  --> AT/SLEW/SLACK LABELS
```

### `*.synthesis_preroute.v` — the netlist, source of **graph nodes**

A flat Verilog gate-level netlist. One module, ports, a few thousand `wire`
declarations, then one instance line per cell:

```verilog
module spm (VGND, VPWR, clk, p, rst, y, x);
 input VGND;  input VPWR;  input clk;  output p;  input rst;  input y;
 input [31:0] x;
 wire _000_;  wire _001_;  ...
```

This is the source of **graph nodes: one node per signal pin.** For every logic
instance, every non-power pin becomes a node named `<instance>/<pin>`; every
primary I/O *bit* becomes a node named after the port bit (`x[0]`, `x[1]`, ...).

**The filename is misleading.** "synthesis_preroute" is just the OpenLane
intermediate stage label. The content is the post-CTS netlist that was placed and
routed — it contains `clkbuf_*` cells inserted by clock tree synthesis, `FILLER_*`
instances inserted by the filler step, and `diode_*` antenna cells. It is not a
raw synthesis output.

### `20-*.def` — routed layout, source of **pin coordinates**

```
VERSION 5.8 ;
DESIGN spm ;
UNITS DISTANCE MICRONS 1000 ;
DIEAREA ( 0 0 ) ( 92740 103460 ) ;
ROW ROW_0 unithd 5520 10880 N DO 177 BY 1 STEP 460 0 ;
...
COMPONENTS 1028 ;
```

All integers are in database units; divide by `UNITS DISTANCE MICRONS` (1000) to
get microns. The `COMPONENTS` section gives each instance's placed origin; the
`PINS` section gives each primary I/O's location.

**This is the only source of geometry**, and it feeds two model inputs:

- `nf[2:6]` — each pin's distance to the die top / left / right / bottom edge
- `net_out` / `net_in` edge feature `ef = (dx, dy)` — the vector from driver pin
  to sink pin, in microns

Those are the *pre-route* inputs. The model is being asked: given where things
sit, how long will the wire between them turn out to be?

### `20-*.spef` — parasitics, feeds the timing engine

```
*SPEF "ieee 1481-1999"
*DESIGN "spm"
*VENDOR "OpenRCX"
*PROGRAM "Parallel Extraction"
*T_UNIT 1 NS   *C_UNIT 1 PF   *R_UNIT 1 OHM
*NAME_MAP
*3 clk
*4 p
```

Per-net resistance and capacitance extracted from the actual routed geometry by
OpenROAD's OpenRCX. **The graph builder never reads this file directly.** Its
role is upstream: it is what makes wire delay *exist* in the SDF. Without SPEF
there are no `INTERCONNECT` delays, and with no interconnect delays there is no
prediction task at all.

### `*.cts_1.sdc` — the constraints

```tcl
create_clock -name clk -period 10.0000 [get_ports {clk}]
set_propagated_clock [get_clocks {clk}]
set_input_delay 2.0000 -clock [get_clocks {clk}] -add_delay [get_ports {rst}]
set_input_delay 2.0000 -clock [get_clocks {clk}] -add_delay [get_ports {x[0]}]
...
set_load -pin_load 0.0177 [get_ports {p}]
set_driving_cell -lib_cell sky130_fd_sc_hd__inv_8 -pin {Y} ... [get_ports {clk}]
set_max_fanout 5.0000 [current_design]
```

Clock period, a uniform 2 ns input delay on every input port, an `inv_8` assumed
as the off-chip driver, 17.7 fF assumed as the off-chip load, max fanout 5.
`set_propagated_clock` means clock latency is computed through the real clock
tree rather than idealised. Remember the `set_input_delay 2.0` — it reappears in
§8 as the thing that distinguishes two different sets of arrival-time labels.

### `*.opensta.tcl` — the most explanatory file in the whole dataset

Ten lines, and it tells you how everything above becomes everything below:

```tcl
set_cmd_units -time ns -capacitance pF -current mA -voltage V -resistance kOhm -distance um
read_liberty -min ../techlib/sky130_fd_sc_hd__ff_n40C_1v95.lib
read_liberty -max ../techlib/sky130_fd_sc_hd__ss_100C_1v60.lib
read_verilog spm.synthesis_preroute.v
link_design spm
read_spef 20-spm.spef
read_sdc spm.cts_1.sdc
report_wns
report_tns
write_sdf -digits 20 spm.at.sdf
exit
```

Two details worth internalising:

1. **Both corners are loaded into one OpenSTA session** (`-min` fast, `-max`
   slow). That is why every number in the SDF is a genuine `min::max` pair — one
   value spanning both corners. A modern LibreLane run instead writes one SDF per
   corner where `min == max`; layer C handles that case explicitly, in
   `build_graph.py:parse_sdf_pair`.
2. **`report_wns` / `report_tns` print to stdout and are never saved.** Worst and
   total negative slack — the headline design-quality numbers — exist nowhere in
   the dataset. Layer C's `openlane/README.md` calls this out: they are "the only
   external check" on a re-derived timing computation.

### `*.sdf` — delay annotation, the **delay features and labels**

Standard Delay Format 3.0, written by OpenSTA (`VENDOR "Parallax"`),
`TIMESCALE 1ns`, 20 decimal digits. Two record kinds matter:

```
(IOPATH A X (0.16062490681065222375::0.38202336819692783720)
            (0.12988714638146664493::0.40426707570244646561))
```

A **cell arc**: pin `A` to pin `X` of the enclosing `(INSTANCE ...)`. The two
parenthesised triples are `(rise_min::rise_max) (fall_min::fall_max)` — four
numbers, which is the project's 4-vector in SDF's own ordering.

```
(INTERCONNECT clk clkbuf_0_clk/A (0.01568915354540295973::0.03149692401600098085)
                                 (0.00476386912152157362::0.01411766403462385959))
```

A **net arc**: driver pin to sink pin, same four numbers. This is the wire delay —
the thing that does not exist before routing.

For `spm`, the file holds **1,553 INTERCONNECT and 700 IOPATH** records across
322 `(CELL ...)` blocks, plus 128 `SETUP`, 128 `HOLD`, 64 `RECOVERY` and
64 `REMOVAL` constraints inside 64 `TIMINGCHECK` blocks.

Many `INTERCONNECT` records are power-net entries with zero delay
(`VGND -> FILLER_0_173/VGND`); they are correctly discarded (§8).

### `*.at.sdf` — the **label file**, and the origin of the name `8_rat`

This is the same SDF *plus* one extra block per design. Verified side by side on
`spm`:

| record | `spm.sdf` | `spm.at.sdf` |
|---|---|---|
| `INTERCONNECT` | 1,553 | 1,553 |
| `IOPATH` | 700 | 700 |
| `SETUP` / `HOLD` | 128 / 128 | 128 / 128 |
| `AT` | — | **2,039** |
| `SLEW` | — | **2,039** |
| `RAT` | — | **2,039** |
| total lines | 5,354 | 11,473 |

The delay records are identical. What `.at.sdf` adds is an `(ARRIVALTIMES)` block
holding, per pin, three quantities in the same
`(rise_min::rise_max) (fall_min::fall_max)` form:

```
 (ARRIVALTIMES
  (AT   clk (0.00000000000000000000::0.00000000000000000000) (5.00000011102230601523::5.00000011102230601523))
  (SLEW clk (0.02567455348771576981::0.05467342705540313014) (0.00816129045083826762::0.02465979573672126210))
  (RAT  clk (-0.14729655412298539585::6.31418890600729465490) (-1e39::1e39))
  (AT   p   (0.58568290763865660242::1.61445361649054275333) (0.59313694523680449144::1.68581053888117704531))
  (SLEW p   (0.08247283645004400976::0.16882370979035768443) (0.05371523863856011299::0.13921077185038321922))
  (RAT  p   (-2.00000000000000000000::8.00000000000000000000) (-2.00000000000000000000::8.00000000000000000000))
  (AT   rst (2.00000000000000000000::2.00000000000000000000) (2.00000000000000000000::2.00000000000000000000))
  ...
```

- **`AT`** = arrival time — when the signal actually gets there
- **`SLEW`** = transition time — how fast the edge is
- **`RAT`** = **required** arrival time — when it needed to get there

**`slack = RAT - AT`.** This is the paper's actual target, and it is why the
derived data directory is named **`data/8_rat`** — "rat" is RAT, not a rodent.

Two details that explain code elsewhere:

- `AT rst = 2.0` exactly matches `set_input_delay 2.0000` in the SDC, and
  `AT clk = 0::0 / 5::5` is the rise at 0 and the fall at 5 of a 10 ns clock.
  These are **absolute** STA arrival times, with the SDC constraints and real
  clock-tree latency applied.
- Pins with no timing requirement carry a sentinel of
  `±1000000043329398871556523614962693701632`, i.e. **±1e39**. That is precisely
  what `data_graph.py` masks out with

  ```python
  invalid_nodes = torch.abs(g.ndata['n_ats']) > 1e20   # ignore all uninitialized stray pins
  ```

### `techlib/` — the library, shared by all 21 designs

| file | gives |
|---|---|
| `merged_unpadded.lef` (2.3 MB) | per-macro `SIZE` and per-pin `DIRECTION` (INPUT/OUTPUT) — this is how the builder knows which pin of a cell is a driver |
| `sky130_fd_sc_hd__ff_n40C_1v95.lib` (71.3 MB) | the `E` corner: pin capacitances to `nf[6:8]`, and the 7x7 delay/transition LUTs |
| `sky130_fd_sc_hd__ss_100C_1v60.lib` (12.8 MB) | the `L` corner: the same, to `nf[8:10]` and the other half of the LUTs |

---

## 5. The derived artifacts — what the model actually eats

### `data/8_rat/<design>.graph.bin` — 21 DGL heterographs, 2.2 GB

This is the only thing `data_graph.py` reads. Sizes range from
`spm.graph.bin` 1.6 MB to `aes256.graph.bin` 430 MB.

Verified schema, by loading `spm.graph.bin` (1,121 nodes, 765 net edges,
700 cell edges):

**Three edge types over a single node set `_N`:**

| edge type | direction | count (spm) | purpose |
|---|---|---|---|
| `net_out` | driver pin → sink pin | 765 | forward signal flow along a wire |
| `net_in` | sink pin → driver pin | 765 | the reverse view, so a driver can aggregate all its sinks |
| `cell_out` | cell input pin → cell output pin | 700 | forward signal flow through a gate |

`net_in` is not redundant: the model needs messages to flow *both* ways across a
net (a driver's delay depends on its total load, i.e. on all its sinks) while
cell arcs only ever propagate forward.

**Node data (`ndata`):**

| tensor | shape | dtype | meaning |
|---|---|---|---|
| `nf` | (n, 10) | float32 | the input node feature |
| `n_net_delays` | (n, 4) | float32 | net arc delay, stored on the *sink* pin |
| `n_ats` | (n, 4) | float32 | arrival time label |
| `n_slews` | (n, 4) | float32 | slew label |
| `n_is_timing_endpt` | (n, 1) | float32 | 1.0 if this pin is a timing endpoint |

The 10 columns of `nf`, with measured ranges on spm:

| col | meaning | spm range |
|---|---|---|
| 0 | `is_primary_IO` | 0 .. 1 |
| 1 | `is_fanout` / is a driver pin | 0 .. 1 |
| 2 | distance to die **top** (µm) | 2.00 .. 101.46 |
| 3 | distance to die **left** (µm) | 0.23 .. 92.23 |
| 4 | distance to die **right** (µm) | 0.51 .. 92.51 |
| 5 | distance to die **bottom** (µm) | 2.00 .. 101.46 |
| 6 | pin capacitance, E corner, rise (pF) | 0 .. 0.00796 |
| 7 | pin capacitance, E corner, fall (pF) | 0 .. 0.00700 |
| 8 | pin capacitance, L corner, rise (pF) | 0 .. 0.00789 |
| 9 | pin capacitance, L corner, fall (pF) | 0 .. 0.00701 |

Columns 2–5 are the DEF geometry; 6–9 are the Liberty capacitances, and only
*fanin* (sink) pins carry one — a driver pin and a primary output both get zero,
since their load is elsewhere or off-chip.

**Edge data (`edata`):**

| edge type | tensor | shape | meaning |
|---|---|---|---|
| `net_out` | `ef` | (e, 2) | `(dx, dy)` driver → sink, in µm |
| `net_in` | `ef` | (e, 2) | the same, negated |
| `cell_out` | `ef` | (e, **512**) | the library timing tables for this arc |
| `cell_out` | `e_cell_delays` | (e, 4) | cell arc delay label |

### Decoding the 512-wide cell edge feature

This is the cleverest part of the data format, and it is worth spelling out.
Each cell arc carries its own **library lookup tables**, verbatim, as a feature:

```
512 = 120 axes + 392 values

120 = 2 corners x 4 table kinds x (1 valid-flag + index_1[7] + index_2[7])
    = 2 x 4 x 15
392 = 2 corners x 4 table kinds x 7 x 7
    = 8 x 49
```

The 4 table kinds, in order: `cell_rise`, `cell_fall`, `rise_transition`,
`fall_transition`. The nesting is `for corner in (E, L): for kind in (...)`, so
slot index = `corner_index * 4 + kind_index`, giving 8 slots.

This is exactly what `model.py:51` relies on when the GCNII baseline does
`g_hetero.edges['cell_out'].data['ef'][:, 120:512].reshape(len(ca), 2*4, 49)` —
skip the axes, reshape the values into 8 tables of 49.

On spm, 95.4 % of the 512 × 700 entries are non-zero, i.e. nearly every arc has
all 8 tables populated.

**Why carry the tables rather than a cell-type embedding?** Because the model
then performs a *differentiable table lookup* (§7) instead of memorising cell
names. That is what makes it a "timing engine inspired" model rather than a
generic GNN, and it is also what lets it generalise to cells it saw rarely.

### `cache/liberty.pkl` — 6.0 MB

A pickle of the fully parsed Liberty data for both corners:
`{corner: {'caps', 'arcs', 'checks', 'senses'}}`. It exists so the 84 MB of
`.lib` text is parsed once instead of once per design — `liberty_lut.py` prints
its own timing when it builds it.

### `checkpoints/` — 2.7 MB, three short local runs

| dir | contents |
|---|---|
| `smoke1/` | `0.pth` (934 KB) + `stdout.log` + empty `stderr.log` |
| `smoke2/` | `0.pth` + two empty logs |
| `pathtest/` | `1.pth` |

`smoke1/stdout.log` is the only training evidence in the folder. It shows a
7-design subset (5 train / 2 test) running 60 epochs:

```
Epoch 0,  net delay 1.763599/1.467313, cell delay 1.370297/0.765512, at 23.867601/(27.703543, 27.682204)
Epoch 19, net delay 0.832963/0.871788, cell delay 0.082238/0.077689, at  4.363369/( 8.921763,  8.992872)
Epoch 39, net delay 0.457500/0.484960, cell delay 0.036366/0.035072, at  3.491184/( 8.531427,  8.626093)
Epoch 59, net delay 0.284923/0.434206, cell delay 0.033673/0.032056, at  3.228481/( 7.691463,  7.799359)
```

Losses fall steadily, which means the pipeline is wired correctly end to end.
The R² values printed at epoch 0 are hugely negative (−23 to −162), as expected
one epoch in. **Nothing here is trained to convergence** — the paper's run goes
to iteration 15,799.

The model is small: 934 KB of float32 ≈ **230k parameters**.

**The paper's pretrained weights are absent.** `README.md` says to place them at
`checkpoints/08_atcd_specul/15799.pth`; that directory does not exist, so the
README's one-command reproduction cannot run in this checkout as-is.

### `.venv312/` — the environment, and why it is pinned

A Python **3.12** virtualenv holding `torch`, `dgl 2.2.1`, `numpy 1.26.4`,
`pandas`, `scikit-learn`, `networkx`, `sympy`, MKL. DGL has no wheels above
Python 3.12, which is the entire reason this venv is separate and version-locked.
`__pycache__/` shows the split in practice: `data_graph`, `model`, `device`,
`liberty_lut`, `tee` compiled for **3.12** (they need torch/dgl), while
`build_graph` and `sta_ref` have **both 3.12 and 3.14** caches — they are
stdlib-only by design and run under either.

---

## 6. Why are there so many Python files?

Seventeen `.py` files at the top level plus four in `openlane/`. The honest
answer is that they belong to **three different generations**, and only four of
them are on the path to the paper's headline result.

### (a) Upstream, and load-bearing — 4 files

| file | role |
|---|---|
| `data_graph.py` | **the data reader.** Loads all 21 `.graph.bin`, derives `n_net_delays_log = log(1e-4 + d) + 7.6`, masks the ±1e39 stray pins, builds `n_atslew = concat(n_ats, log(1e-4 + n_slews) + 3)` (8 channels), computes the topological level list, and pre-computes the node-role index sets (`input_nodes`, `output_nodes`, `pi_nodes`, `po_nodes`, `endpoints`) |
| `model.py` | **the model.** `MLP`, `NetConv`, `SignalProp`, `TimingGCN`, plus the `AllConv` / `DeepGCNII` baseline |
| `train_gnn.py` | **train and test.** `--test_iter` to score a saved model, `--checkpoint` to name a run, `--no_netdelay` / `--no_celldelay` / `--no_groundtruth` ablations |
| `tee.py` | mirrors stdout/stderr into the checkpoint directory |

The train/test split lives in `data_graph.py` and is fixed by construction:

```python
random.seed(8026728)
available_data = 'blabla usb_cdc_core BM64 jpeg_encoder salsa20 ... usb'.split()
train_data_keys = random.sample(available_data, 14)
```

**14 train / 7 test**, reproducible across machines because the seed and the list
order are both fixed. Anything *not* in `train_data_keys` lands in `data_test` —
a property layer C exploits deliberately (see `TIMINGPREDICT_EXTRA` below).

### (b) Upstream baselines and side experiments — 6 files, all dead in this checkout

| files | baseline | needs | present? |
|---|---|---|---|
| `train_stat.py` + `data_stat.py` | plain MLP on per-net statistics | `data/4_netstat/*.netstat.pt` | **no** |
| `train_stat_rf.py` + `data_stat_cpu.py` | `RandomForestRegressor` on the same | `data/4_netstat/` + `checkpoints/netstat_rf.pickle` | **no** |
| `train_deepgnn.py` + `data_homograph.py` | 60-layer GCNII on a homogeneous graph | `data/7_homotest/train_test.pt` | **no** |
| `plot_slacks.py` | the paper's slack scatter plot | `checkpoints/18_slacksdump/*.npz` | **no** |

None of those directories was ever downloaded, so **these six files cannot run
here at all.** They are the single biggest reason the file count looks inflated:
they are the paper's comparison baselines, not part of the main pipeline. They
also still contain hardcoded `.cuda()` calls and the Python ≤ 3.10
`random.sample(dict.items(), k)` idiom, so they would need the same treatment
layer C gave the four core files.

The numbering in those paths (`4_netstat`, `7_homotest`, `8_rat`,
`18_slacksdump`) is the authors' own experiment log — `8_rat` was experiment 8,
the one that became the paper.

### (c) Added locally, 13–16 Oct 2025 — 5 files + `openlane/`

This is the substantive local work: **a complete, independent reimplementation of
the dataset build**, so the graphs can be regenerated from `netlists/` rather
than downloaded, and so new designs can be added.

| file | job |
|---|---|
| `device.py` | 750 B. One place that decides CPU vs CUDA, honouring `TIMINGPREDICT_DEVICE`, plus a `sync()` that is a no-op off-GPU so timings stay honest. Exists because upstream hardcodes `.cuda()` and DGL ships no CUDA build for sm_120 / RTX 50-series |
| `build_graph.py` | 19.8 KB, **stdlib only, no torch/dgl**. Parses the LEF (macro sizes, pin directions), the Verilog (instances, port bits, `assign` aliases), the SDF (`IOPATH` + `INTERCONNECT`) and the DEF (die, instance and port coordinates), and assembles the pin DAG. Also has `--dir` mode and `parse_sdf_pair` for one-SDF-per-corner flows |
| `liberty_lut.py` | 9.9 KB. A brace-depth-tracking Liberty parser producing per-arc 7x7 LUTs, pin capacitances, constraint types and timing senses, cached to `cache/liberty.pkl` |
| `sta_ref.py` | 4.9 KB. A pure-Python levelized forward static timing analysis (Kahn topological order, CSR-style linked adjacency) over the SDF-annotated DAG, producing arrival times |
| `make_dataset.py` | 10.8 KB. Assembles all of the above into the DGL heterograph and writes `data/8_rat/<design>.graph.bin` |
| `openlane/` | 48 KB. `README.md`, `harden.sh`, `fix_sv.py`, `mkconfig.py`, `sta_dataset_style.tcl`, and `tiny_gpu/config.ol{1,2}.json` + a wrapper RTL file |

Several design decisions in there are worth noting because they show real care:

- `build_graph.py:FILE_PATTERNS` locates each file **by extension, most specific
  pattern first, largest match within a pattern** — because the routed DEF is
  bigger than the floorplan DEF and the powered netlist bigger than the bare one.
  That is what makes `--dir` work on a fresh OpenLane run whose stage names
  differ from the dataset's.
- Conditioned `IOPATH` records (several per pin pair, one per `when` condition)
  are collapsed into one arc holding the **worst-case envelope**
  (`_merge_worst`: min of mins, max of maxes).
- `make_dataset.py` builds the 512-wide feature **once per distinct
  `(master, from_pin, to_pin)`** and indexes it out per edge — the comment notes
  a per-edge Python loop would be "100x the work".
- `sta_ref.py` handles `timing_sense` correctly: a `negative_unate` arc swaps
  rise and fall, a `non_unate` arc takes whichever input edge is worse.

### `openlane/` — hardening a GPU as a 22nd design

The reasoning in `openlane/README.md` is sound and worth preserving here:
three of the GNN's inputs **do not exist until the design is placed and routed**
— `nf[2:6]` and the `(dx, dy)` edge feature need DEF placement, and
`n_net_delays` needs SDF `INTERCONNECT`, which needs SPEF. So synthesis alone is
useless; the flow has to run to detailed routing and extraction, which means
OpenROAD and OpenSTA, which means Linux.

The target chosen is [adam-maj/tiny-gpu](https://github.com/adam-maj/tiny-gpu):
12 SystemVerilog files, MIT, a real SIMT machine (dispatcher, N cores, each with
scheduler / fetcher / decoder / per-thread PC, register file, ALU, LSU). The
reason it is the right first target is what it *lacks* — both memories are
off-chip ports, so there is no SRAM macro and no blackbox, making it a pure
standard-cell design like all 21 dataset circuits.

Measured (not guessed) at `NUM_CORES=2, THREADS_PER_BLOCK=4`: 13,018
`sky130_fd_sc_hd` cells, 2,283 flip-flops (1,259 `dfxtp_1` + 1,024 `edfxtp_1` —
the 1,024 are exactly the register files, 16 regs x 8 bits x 8 threads),
122,564 µm² cell area, 18 top-level ports / 183 bits. The README extrapolates
~46,000 graph nodes from the dataset's steady 3.4–3.9 pins per logic cell, which
would place it between `genericfir` and `y_huff`.

The two real obstacles, both diagnosed and solved:

1. **Trailing commas** before the `)` of a parameter list — legal in Icarus,
   illegal SystemVerilog, rejected by Verilator and slang. `fix_sv.py` removes
   exactly seven of them, and only where a newline separates the comma from the
   `)`, since `u_foo(a, b, )` on one line is a legal omitted port connection.
2. **Unpacked array ports** (`input reg [7:0] x [3:0]`), which Yosys's Verilog
   frontend cannot parse in *any* module. `sv2v` resolves both by lowering to
   Verilog-2005.

`sta_dataset_style.tcl` is a behavioural copy of the dataset's own
`<design>.opensta.tcl` with filenames taken from the environment, so a newly
hardened design gets an SDF in exactly the dataset's format.

### The local edits to upstream — small and surgical

`git diff` is **4 files, +56 / −19**:

| change | why |
|---|---|
| `.cuda()` → `device.py`'s `DEV`, `torch.cuda.synchronize()` → `sync()`, `device='cuda'` → `device=nf.device` | run on CPU-only or non-CUDA-DGL machines |
| node id tensors `int32` → `int64` | DGL ≥ 2 rejects int32 node ids |
| `random.sample(data_train.items(), k)` → `random.sample(list(...), k)` | Python ≥ 3.11 requires a sequence |
| `batch_size = 7` → `min(env TIMINGPREDICT_BATCH, len(data_train))` | `random.sample` raises if the batch exceeds the train split |
| new `TIMINGPREDICT_DESIGNS`, `TIMINGPREDICT_EXTRA`, `TIMINGPREDICT_BATCH`, `TIMINGPREDICT_EPOCHS` env vars | all 21 graphs resident at once costs several GB, more than a 16 GB laptop has free |

`TIMINGPREDICT_EXTRA` is the clever one. The split is drawn from the *original*
21-name list before any extras are appended, so a newly hardened design
**necessarily lands in `data_test`** — which is the point: harden a new circuit,
then score a model that provably never saw it.

---

## 7. The model — why it is called "timing engine inspired"

`TimingGCN` has two stages, and they mirror what a real STA engine does.

```python
class TimingGCN(torch.nn.Module):
    def __init__(self):
        self.nc1  = NetConv(10, 2, 32)
        self.nc2  = NetConv(32, 2, 32)
        self.nc3  = NetConv(32, 2, 16)   # 16 = 4x delay + 12x arbitrary
        self.prop = SignalProp(10 + 16, 8, 7, 8, 4)

    def forward(self, g, ts, groundtruth=False):
        nf0 = g.ndata['nf']                      # (n, 10)
        x = self.nc3(g, ts, self.nc2(g, ts, self.nc1(g, ts, nf0)))   # (n, 16)
        net_delays = x[:, :4]                    # <-- prediction 1
        nf1 = torch.cat([nf0, x], dim=1)         # (n, 26)
        nf2, cell_delays = self.prop(g, ts, nf1, groundtruth=groundtruth)
        return net_delays, cell_delays, nf2      # <-- predictions 2 and 3
```

### Stage 1 — `NetConv` x 3: local, parallel, net-level

Three message-passing layers over the net edges only, 10 → 32 → 32 → 16.
Each layer does two different things depending on which end of a net a pin is on:

- **sink pins** aggregate (sum) a message from their driver, along `net_out`
- **driver pins** aggregate from *all* their sinks, along `net_in`, with **both**
  `sum` and `max` pooling, then a reduce MLP

The sum/max pair matters physically: total load is a sum, while the critical sink
is a max. The message is also gated — the MLP emits an extra scalar `k`, passed
through `sigmoid`, that scales the features (`f1 * k, f2 * k`), letting the layer
suppress irrelevant neighbours.

**The first 4 of the 16 output channels *are* the net delay prediction.** The
remaining 12 are free latent channels (the code comment guesses "might include
cap, beta"). So net delay is predicted purely locally, in parallel, in 3 hops —
which is correct, because wire delay genuinely is a local property of a net and
its geometry.

### Stage 2 — `SignalProp`: a learned levelized forward propagation

This is the paper's central idea. A real STA engine computes arrival times by
walking the circuit in topological order, level by level, relaxing arrival times
along each arc. `SignalProp` does exactly that, but every "relax" step is a
learned function:

```python
assert len(ts['topo']) % 2 == 0, 'The number of logic levels must be even (net, cell, net)'
...
for i in range(1, len(ts['topo'])):
    if i % 2 == 1: prop_net(ts['topo'][i], groundtruth)    # across a wire
    else:          prop_cell(ts['topo'][i], groundtruth)   # through a gate
```

Levels strictly alternate net → cell → net → cell, because the graph alternates
by construction (a net arc always lands on a cell input, a cell arc always lands
on a cell output). Primary inputs are seeded with ground-truth
`n_atslew` (`node_skip_level_o`), which is the equivalent of an STA engine's
start points.

This is also why `make_dataset.py` records `levels_even` and warns on an odd
count — an odd number would break the assertion.

### The differentiable LUT lookup

Inside `prop_cell`, `SignalProp.edge_msg_cell` is where the 512-wide feature
earns its existence. A real STA engine, for each cell arc, interpolates the
Liberty table at (input slew, output load). This does the learned analogue:

1. **Form a query.** An MLP reads the arc's current state (propagated
   `new_nf` of the source pin, plus both pins' `nf`) and emits
   `8 luts x 4 duplicates x 2` numbers — 32 two-dimensional queries, i.e. 32
   guesses at "where on the table should I look?" (`lut_dup=4` gives the layer
   four independent probes per table).
2. **Answer it against the real axes.** `MLP_lut_attention` takes each query
   together with that table's actual `index_1[7]` / `index_2[7]` axis values and
   the valid flag, and emits 7 + 7 attention weights — a soft position along each
   axis.
3. **Build a 7x7 mask.** `torch.matmul(ax, ay)` as a batched outer product turns
   the two 7-vectors into a 7x7 weight matrix.
4. **Read the table.** A batched dot product of that mask with the flattened 49
   table values returns one scalar per probe — a soft, differentiable table
   lookup. 32 scalars `r` come back per edge.
5. **Emit the message.** `MLP_cellarc_msg` consumes `[state, src nf, dst nf, r]`
   and emits a sigmoid gate `k`, two 32-wide feature halves, and a 4-wide
   `efce` — **which is the cell delay prediction for that arc.**
6. Sum and max pooling over all arcs entering an output pin, then
   `MLP_cellreduce` → the pin's 8-wide `new_nf`.

So the model does not memorise "an `and2_1` is 0.14 ns". It learns *how to read a
datasheet*, and the datasheet travels with the edge. That is the sense in which
it is "timing engine inspired", and it is why the authors expect it to handle
cell types it rarely saw.

### What the 8 output channels are

`new_nf` is 8 wide, matching `n_atslew = concat(n_ats, log(1e-4 + n_slews) + 3)`:

| channels | target |
|---|---|
| 0:4 | arrival time `[E_rise, E_fall, L_rise, L_fall]` |
| 4:8 | log-scaled slew, same ordering |

`train_gnn.py:test()` scores only `[:, :4]` — the arrival times.

### Training

```python
loss = MSE(pred_net_delays, n_net_delays_log)      # log(1e-4 + d) + 7.6
     + MSE(pred_cell_delays, e_cell_delays)
     + MSE(pred_atslew, n_atslew)
```

Adam, `lr=5e-4`, 7 designs sampled per epoch, up to 100,000 epochs,
gradients accumulated across the 7 then one `optimizer.step()`.

Two mechanisms worth knowing:

- **Log scaling of net delay.** `log(1e-4 + d) + 7.6` turns a heavily
  right-skewed, near-zero-heavy delay distribution into something a regressor can
  fit. The `+7.6` recentres it (`log(1e-4) ≈ -9.21`), and `+3` plays the same
  role for slews.
- **Teacher forcing (`groundtruth=True`, the default).** In this mode
  `SignalProp` feeds each arc the *true* `n_atslew` of its source pin instead of
  the propagated prediction, so no iteration is needed and errors cannot
  compound during training. Testing always uses `groundtruth=False` — full
  propagation from the primary inputs, which is the honest setting. The training
  log prints both (`at 3.23/(7.69, 7.79)` = train / test-teacher-forced /
  test-propagated), so you can watch the gap.
- **A deliberate dgl workaround**: when `--no_celldelay` is passed, the code
  still computes `torch.sum(pred_cell_delays) * 0.0` to keep the gradient channel
  alive, because dropping an unused forward branch made GPU memory "BOOM".

### The known R² typo, deliberately left in

```python
# notice: there is a typo in the parameter order of r2 calculator.
# please see https://github.com/TimingPredict/TimingPredict/issues/7.
# for exact reproducibility of experiments in paper, we will not directly fix the typo here.
r2 = r2_score(pred.cpu().numpy().reshape(-1), truth.cpu().numpy().reshape(-1))
```

`sklearn.metrics.r2_score(y_true, y_pred)` — the arguments are swapped here, so
the reported value is R² of truth-against-prediction rather than the reverse.
The authors keep it so published numbers reproduce. Commit `b24db2c` in this
repo's history is titled "r2 calculation error notice". **If you compare your own
numbers against the paper, keep the typo; if you want a correct R², swap them and
say so.**

### The `DeepGCNII` baseline, for contrast

Also in `model.py`: `AllConv` + `DeepGCNII`, a 60-layer GCNII with residual
connections and initial-feature re-injection, operating on a *homogeneous*
bidirectional graph (`gen_homobigraph_with_features` flattens the three edge
types into one and mean-pools the 8 cell tables down to 8 numbers). It is the
"just stack more GNN layers instead of respecting the timing graph" strawman, and
it needs `data/7_homotest/` which is absent here.

---

## 8. Verified status of this checkout, and the gaps that matter

Everything in this section was checked during this analysis, not inferred.

### What works, confirmed

**Structural reproduction is exact.** Running the local builder on seven designs:

```
$ .venv312/Scripts/python.exe -I build_graph.py netlists \
      spm usb cic_decimator zipdiv xtea wbqspiflash usb_cdc_core

design             nodes    net_e   cell_e   endpts   vs reference
spm                 1121      765      700      129   EXACT
usb                 3361     2406     2189      344   EXACT
cic_decimator       3131     2232     2102      130   EXACT
zipdiv              4398     3102     2913      181   EXACT
xtea               10213     7151     6882      423   EXACT
wbqspiflash         9672     6798     6454      323   EXACT
usb_cdc_core        7406     5200     4869      630   EXACT
```

`EXACT` on all four counts for all seven. That is a strong result: node
identification, pin-direction resolution, `assign` alias chasing, physical-cell
filtering, conditioned-arc collapsing and endpoint identification all match the
published dataset exactly.

The "unmapped net" counts it also prints (e.g. 788 for spm) are the
`VGND`/`VPWR` → filler-cell `INTERCONNECT` records; dropping them is correct, and
the raw totals it reports (spm: 700 IOPATH, 1,553 INTERCONNECT) match the SDF.

The graphs also load and train: `checkpoints/smoke1/stdout.log` shows 60 clean
epochs with falling losses.

### What differs from the published dataset

**`data/8_rat` in this folder was generated locally, not downloaded.** The
mtimes say so: `cache/liberty.pkl` at 13 Oct 18:39, then all 21 `.graph.bin`
between 18:42 and 18:43. So the labels come from `sta_ref.py`, **not** from the
`(ARRIVALTIMES)` block of the `.at.sdf` files sitting right there on disk.

Two concrete consequences, both measured on `spm.graph.bin`:

**1. `n_slews` is entirely zero.** 0 of 4,484 entries are non-zero. `sta_ref.py`
is honest about it in its own docstring:

> Slews are not recoverable from SDF (it carries no transition times) and are
> returned as zeros.

That premise is wrong for `.at.sdf` specifically — it carries 2,039 `(SLEW ...)`
records per design (§4). The effect: `n_atslew` channels 4:8 become the constant
`log(1e-4 + 0) + 3 ≈ -6.21`, so **half of the 8 regression channels are
degenerate**. It does not break training (the reported R² uses `[:, :4]`), but a
quarter of the loss term is learning a constant.

**2. `n_ats` is cone-relative, not absolute.** Measured range for spm:
0.0 to 2.774 ns. `sta_ref.py` starts every primary input *and every sequential
output* at 0, and applies no clock latency and no SDC input delay — again stated
plainly in its docstring ("cone-relative arrivals, not slacks"). The `.at.sdf`
by contrast has absolute STA arrivals: `AT rst = 2.0` (matching
`set_input_delay 2.0000`) and `AT clk = 0::0 / 5::5` for a 10 ns clock.

Also, zero pins hit the ±1e39 sentinel in the locally built spm graph, whereas
the published format expects strays (`data_graph.py` masks them) — another sign
these labels were computed differently.

**None of this makes the checkout broken.** It is internally consistent: train
here and score here and the numbers mean something. But they are **not the
paper's numbers**, and `openlane/README.md` already draws the matching
conclusion for the cell-feature layout:

> The 512-wide cell feature's layout here is inferred, not verified against the
> authors' graphs, so a GPU graph works with a **self-trained** checkpoint. The
> pretrained `15799.pth` is not expected to transfer to it.

### The cheap, high-value fix

Everything needed for real labels is already on disk. `build_graph.py:parse_sdf`
currently matches only two record types:

```python
elif s.startswith('(IOPATH'):        ...
elif s.startswith('(INTERCONNECT'):  ...
```

Adding `(AT`, `(SLEW` and `(RAT` — same `_quad` triple parser, keyed by pin name
instead of by instance — would deliver, with no new tooling and no re-running any
EDA flow:

- **real slews**, rescuing 4 of the 8 regression channels
- **absolute arrival times** including input delays and clock-tree latency
- **slack**, as `RAT - AT`, which is the quantity in the paper's title and which
  nothing in this checkout currently computes

`sta_ref.py` would then become what its own docstring says it is — a *reference*
implementation to check the parsed labels against, and the CPU baseline for a GPU
timing kernel — rather than the source of the labels themselves.

### What is not present or not yet run

| missing | consequence |
|---|---|
| `checkpoints/08_atcd_specul/15799.pth` | the README's one-command reproduction cannot run |
| `data/4_netstat/` | `train_stat.py`, `train_stat_rf.py`, `data_stat*.py` cannot run |
| `data/7_homotest/` | `train_deepgnn.py`, `data_homograph.py` cannot run |
| `checkpoints/18_slacksdump/` | `plot_slacks.py` cannot run |
| `openlane/hardened/` | the GPU has not been hardened yet; needs Linux / WSL2 / Colab, since OpenROAD and OpenSTA have no Windows build |
| any WNS / TNS reference | `report_wns` / `report_tns` output was never captured, so there is no external check on any arrival-time computation |

---

## 9. Quick reference

### Disk map

| path | size | tracked by git? |
|---|---|---|
| `netlists/` | 3.5 GB | no (`.gitignore`) |
| `netlists/techlib/` | 83 MB | no |
| `data/8_rat/` | 2.2 GB | no |
| `cache/liberty.pkl` | 6.0 MB | no |
| `checkpoints/` | 2.7 MB | no |
| `.venv312/` | — | no |
| `openlane/` | 48 KB | **yes** (minus `hardened/`, `runs/`, `gen/`, upstream RTL) |
| the ~17 `.py` files | ~110 KB | **yes** |

That last row is the other half of the answer to "why so many Python files":
**the Python files are essentially the entire tracked repository.** Everything
large is a local download or a local build product. The `.gitignore` says so
directly, including the locally added lines:

```
# raw dataset download and derived liberty cache
netlists/
cache/

# OpenLane build artifacts: upstream RTL clones, sv2v output, flow runs, results
openlane/*/src/*.sv
!openlane/*/src/tiny_gpu_top.sv
openlane/*/gen/
openlane/*/runs/
openlane/hardened/
```

### The pipeline, end to end

```
netlists/<design>/*.v   ─┐
netlists/<design>/*.sdf ─┤
netlists/<design>/*.def ─┼─> build_graph.py ──> pin DAG (nodes + net/cell arcs)
netlists/techlib/*.lef  ─┘                           │
                                                     │
netlists/techlib/*.lib ──> liberty_lut.py ──> cache/liberty.pkl
                                     │               │
                                     ▼               ▼
                             make_dataset.py <── sta_ref.py (labels)
                                     │
                                     ▼
                     data/8_rat/<design>.graph.bin   (21 files, 2.2 GB)
                                     │
                                     ▼
                              data_graph.py ──> 14 train / 7 test
                                     │
                                     ▼
                       train_gnn.py + model.py (TimingGCN)
                                     │
                                     ▼
                      net delay · cell delay · arrival time
```

### Commands that work in this checkout

Everything needing torch/dgl must use the pinned interpreter, from the repo root:

```bash
# 1. parse both Liberty corners into the cache (only if cache/liberty.pkl is gone)
.venv312/Scripts/python.exe -I liberty_lut.py netlists cache/liberty.pkl

# 2. structural sanity check against the published counts -- stdlib only, fast
.venv312/Scripts/python.exe -I build_graph.py netlists spm usb cic_decimator
.venv312/Scripts/python.exe -I build_graph.py netlists --all        # slow, 3.5 GB

# 3. rebuild the DGL graphs (already done; ~2.2 GB of output)
.venv312/Scripts/python.exe make_dataset.py netlists data/8_rat --all

# 4. train on a laptop-sized subset
TIMINGPREDICT_DESIGNS=spm,usb,cic_decimator,zipdiv,xtea,wbqspiflash,usb_cdc_core \
TIMINGPREDICT_EPOCHS=200 \
TIMINGPREDICT_BATCH=5 \
.venv312/Scripts/python.exe train_gnn.py --checkpoint myrun

# 5. score a saved checkpoint
.venv312/Scripts/python.exe train_gnn.py --test_iter 199 --checkpoint myrun
```

Useful environment variables (all added locally):

| var | effect |
|---|---|
| `TIMINGPREDICT_DEVICE` | `cpu` or `cuda`, overriding auto-detection |
| `TIMINGPREDICT_DESIGNS` | comma-separated subset of the 21 to load |
| `TIMINGPREDICT_EXTRA` | extra graph names to append; they land in `data_test` by construction |
| `TIMINGPREDICT_BATCH` | designs per epoch (default 7) |
| `TIMINGPREDICT_EPOCHS` | epoch cap (default 100,000) |

### The 4-vector cheat sheet

Every timing tensor in the project, in order:

```
index:   0         1         2         3
        E_rise    E_fall    L_rise    L_fall
        |---- ff / fast / -min ----|  |---- ss / slow / -max ----|
         1.95 V, -40 C                 1.60 V, 100 C
```

E arrivals combine with `min`, L arrivals with `max`. SDF's own ordering is
`(rise_min::rise_max) (fall_min::fall_max)`, i.e. `[E_rise, L_rise, E_fall,
L_fall]`, so it is permuted on the way in — `make_dataset.py:canon`.

### Glossary

| term | meaning |
|---|---|
| **STA** | static timing analysis — compute worst-case timing without simulation |
| **AT / RAT** | arrival time / required arrival time; `slack = RAT - AT` |
| **slew** | signal transition time (how fast an edge is) |
| **WNS / TNS** | worst / total negative slack — the headline timing quality numbers |
| **net arc** | driver pin to sink pin, across a wire; delay exists only after routing |
| **cell arc** | input pin to output pin of a gate; delay from the Liberty tables |
| **timing endpoint** | where a path must end: a flop data/control input, or a primary output |
| **LEF / DEF** | library/design exchange format — cell abstracts / the actual layout |
| **SPEF** | extracted wire resistance and capacitance |
| **SDC** | the timing constraints (clocks, I/O delays) |
| **SDF** | the annotated delays, and here also the arrival times |
| **Liberty (`.lib`)** | the characterised cell library: delays, slews, capacitances, per corner |
| **unate** | whether a rising input causes a rising (`positive_unate`) or falling (`negative_unate`) output |
| **CTS** | clock tree synthesis — inserts the `clkbuf_*` cells you see in the netlists |
| **decap / fill / tap** | physical-only cells with no signal pins; dropped from the graph |

---

## 10. Claude's plan of action for this summary

The plan below is the one approved before this document was written, reproduced
verbatim. It doubles as a map of the document: each numbered item under
"Deliverable" corresponds to the section of the same number above.

### Plan: write `my_work/analysis.md` — full DNA analysis of the TimingPredict folder

#### Context

`C:\Users\supre\Downloads\TimingPredict` is a clone of the DAC'22 **TimingPredict**
repo (Guo et al., *A Timing Engine Inspired Graph Neural Network Model for
Pre-Routing Slack Prediction*) plus the raw 3.5 GB **sky130 / OpenLane** dataset
under `netlists/`, plus a second, locally-written ingest pipeline (`build_graph.py`,
`liberty_lut.py`, `sta_ref.py`, `make_dataset.py`, `device.py`, `openlane/`) that a
developer added on 13–16 Oct 2025 to regenerate the binary graphs from raw EDA
files and to extend the dataset with a hardened GPU.

The user wants one document that explains: what the dataset actually is, how the
file types connect, why there are ~17 Python files, what process technology is
involved, and the end-to-end data flow — a "complete DNA analysis". Nothing is
to be modified; the only deliverable is `my_work/analysis.md`.

Everything below is already verified in this session (file reads, greps,
`build_graph.py` run on 7 designs, DGL load of `spm.graph.bin`). No further
exploration is needed before writing.

#### Deliverable

Single file: `C:\Users\supre\Downloads\TimingPredict\my_work\analysis.md`
(currently exists, empty). Markdown, ~600–800 lines, tables where they earn it,
written so an engineer new to the repo can navigate it. Sections in this order:

##### 1. One-page summary
What the folder is: three layers stacked in one directory — (a) the authors'
published GNN code, (b) the raw post-route EDA dataset it was built from, (c) a
local re-implementation of the dataset build + an OpenLane path to add new designs.
The ML task: predict post-route timing (net delay, cell delay, arrival time,
slack) from **pre-route** information, so a designer sees routed-quality timing
at placement time instead of after a multi-hour route.

##### 2. The technology — answering "130nm, 90nm etc."
- **One node only: SkyWater sky130 (130 nm) open-source CMOS PDK.** No 90 nm,
  no mixed nodes. Evidence: every cell master is `sky130_fd_sc_hd__*`;
  `netlists/techlib/merged_unpadded.lef` is SkyWater-copyrighted, 441 MACROs,
  `SITE unithd 0.46 × 2.72 µm`, MANUFACTURINGGRID 0.005 µm.
- Library = `sky130_fd_sc_hd` (high-density standard cells), 132 distinct
  masters used in picorv32a.
- **Two PVT corners only**, and they define the `E`/`L` ("early/late") halves of
  every 4-vector in the whole project:
  | tag | Liberty file | V | T | role |
  |---|---|---|---|---|
  | `E` | `sky130_fd_sc_hd__ff_n40C_1v95.lib` (71 MB) | 1.95 V | −40 °C | fast / `read_liberty -min` |
  | `L` | `sky130_fd_sc_hd__ss_100C_1v60.lib` (12.8 MB) | 1.60 V | 100 °C | slow / `read_liberty -max` |
  `technology("cmos")`, `delay_model : table_lookup`, units ns / pF / V / mA.
- Delay LUT template `del_1_7_7`: 7×7, `index_1 = input_net_transition`,
  `index_2 = total_output_net_capacitance` — which is why the model's cell
  feature is natively 7×7 with no resampling.
- Clock periods are per-design, 10–65 ns (table in §3). Die areas 93×103 µm
  (spm) to 1684×1695 µm (jpeg_encoder).

##### 3. The dataset — 21 designs
Explain it is the **TimingPredict/Dataset** repo content: 21 open-source RTL
blocks each taken through OpenLane → OpenROAD → OpenSTA in sky130, captured at
the *post-route* stage. Table with, per design: clock period, DEF COMPONENTS,
die size, and the published graph counts (nodes / net edges / cell edges /
endpoints) from `build_graph.py:REF`. Verified numbers to use:

| design | clk ns | DEF comps | die µm | nodes | net_e | cell_e | endpts |
|---|---|---|---|---|---|---|---|
| spm | 10.00 | 1,028 | 93×103 | 1,121 | 765 | 700 | 129 |
| cic_decimator | 10.00 | 2,676 | 142×153 | 3,131 | 2,232 | 2,102 | 130 |
| usb | 12.55 | 3,376 | 169×180 | 3,361 | 2,406 | 2,189 | 344 |
| zipdiv | 19.09 | 4,022 | 176×186 | 4,398 | 3,102 | 2,913 | 181 |
| usb_cdc_core | 15.60 | 6,729 | 223×234 | 7,406 | 5,200 | 4,869 | 630 |
| xtea | 26.03 | 8,136 | 236×247 | 10,213 | 7,151 | 6,882 | 423 |
| wbqspiflash | 18.86 | 9,017 | 262×272 | 9,672 | 6,798 | 6,454 | 323 |
| synth_ram | 50.00 | 24,305 | 458×469 | 25,910 | 19,024 | 16,782 | 2,112 |
| genericfir | 10.00 | 34,719 | 549×559 | 38,827 | 28,845 | 25,013 | 3,811 |
| BM64 | 20.00 | 57,983 | 711×721 | 38,458 | 27,843 | 25,334 | 1,800 |
| y_huff | 11.35 | 42,451 | 565×576 | 48,216 | 33,689 | 30,612 | 2,391 |
| picorv32a | 24.73 | 57,366 | 660×670 | 58,676 | 43,047 | 40,208 | 1,920 |
| aes_cipher | 10.09 | 65,607 | 703×713 | 59,777 | 42,671 | 41,411 | 660 |
| des | 20.00 | 59,687 | 683×694 | 60,541 | 44,478 | 41,845 | 2,048 |
| usbf_device | 14.60 | 59,757 | 652×663 | 66,345 | 46,241 | 42,226 | 4,404 |
| blabla | 65.00 | 123,336 | 1056×1067 | 55,568 | 39,853 | 35,689 | 1,614 |
| salsa20 | 36.73 | 121,934 | 1040×1051 | 78,486 | 57,737 | 52,895 | 3,710 |
| aes128 | 11.43 | 238,867 | 1355×1365 | 211,045 | 148,997 | 138,457 | 5,696 |
| aes192 | 12.37 | 327,794 | 1647×1658 | 234,211 | 165,350 | 152,910 | 8,096 |
| jpeg_encoder | 15.78 | 337,006 | 1684×1695 | 238,216 | 176,737 | 167,960 | 4,422 |
| aes256 | 15.00 | 309,677 | 1533×1544 | 290,955 | 207,414 | 189,262 | 11,200 |

Note the composition bias worth flagging: 4 AES variants + aes_cipher + des +
salsa20 + xtea = 8 of 21 are crypto datapaths; 4 are USB; one CPU; no GPU
(the motivation for `openlane/`). Note that DEF COMPONENTS >> cell edges because
~50–70 % of placed instances are `decap`/`fill`/`tap` physical-only cells with no
signal pins — `build_graph.py:PHYS_RE` drops them.

##### 4. The seven raw file types and how they connect
A diagram-style walkthrough keyed to one design directory (`netlists/spm/`),
explaining for each file what it holds and which GNN tensor depends on it:
- `*.synthesis_preroute.v` — flat gate-level netlist. Source of **nodes** (one
  per signal pin) and the instance→master mapping. Note the misleading name: the
  content is the post-CTS/post-route netlist OpenLane calls "preroute".
- `20-*.def` — routed DEF, `UNITS DISTANCE MICRONS 1000`, DIEAREA, COMPONENTS
  with placement, PINS with port locations. Source of **pin coordinates** →
  `nf[2:6]` (distance to die top/left/right/bottom) and net edge `ef = (dx,dy)`.
- `20-*.spef` — OpenRCX-extracted RC parasitics (`*SPEF "ieee 1481-1999"`,
  `*VENDOR "OpenRCX"`). Not read by the graph builder directly; it is the input
  to OpenSTA that makes wire delay exist at all.
- `*.cts_1.sdc` — constraints: `create_clock`, `set_propagated_clock`,
  `set_input_delay 2.0`, `set_driving_cell inv_8`, `set_max_fanout 5`.
- `*.opensta.tcl` — the 10-line recipe that produced the SDF; the single most
  explanatory file in the dataset. Reads `-min` ff and `-max` ss **together**,
  then SPEF, then SDC, then `write_sdf -digits 20`. This is why every SDF triple
  is a genuine `ff::ss` pair rather than two separate files.
- `*.sdf` — the delay file: `(IOPATH from to (r_min::r_max) (f_min::f_max))` cell
  arcs and `(INTERCONNECT src dst ...)` net arcs. **Features/labels for delay.**
- `*.at.sdf` — the same SDF **plus** an `(ARRIVALTIMES)` block per design:
  `(AT pin ...)`, `(SLEW pin ...)`, `(RAT pin ...)` per pin, each as the same
  rise-min::max / fall-min::max pair. **This is the label file**, and it is what
  the directory name `data/8_rat` refers to — RAT = Required Arrival Time;
  `slack = RAT − AT`. Verified on spm: `.sdf` 5,354 lines vs `.at.sdf` 11,473,
  identical 1,553 INTERCONNECT / 700 IOPATH, plus 2,039 each of AT/SLEW/RAT.
  Unconstrained pins carry ±1e39 sentinels — exactly what `data_graph.py` masks
  with `abs(n_ats) > 1e20`.
- `techlib/` — the LEF (pin directions, macro sizes) and the two Liberty corners
  (pin capacitance → `nf[6:10]`, and the 7×7 delay/transition LUTs → the
  512-wide cell edge feature).

##### 5. The derived artifacts
- `data/8_rat/<design>.graph.bin` — 21 DGL heterographs, 2.2 GB. Full tensor
  schema, verified by loading spm:
  - 3 edge types on one node set `_N`: `net_out` (driver→sink), `net_in`
    (reverse, for the net-level message passing), `cell_out` (cell in-pin→out-pin).
  - `ndata`: `nf` (n,10), `n_net_delays` (n,4), `n_ats` (n,4), `n_slews` (n,4),
    `n_is_timing_endpt` (n,1).
  - `edata`: `net_out/net_in ef` (e,2), `cell_out ef` (e,512),
    `cell_out e_cell_delays` (e,4).
  - Every 4-vector is `[E_rise, E_fall, L_rise, L_fall]`.
  - The 512 = 120 axes + 392 values = `2 corners × 4 table kinds ×
    (1 valid flag + index_1[7] + index_2[7])` and `2 × 4 × 49`; table kinds are
    `cell_rise, cell_fall, rise_transition, fall_transition`. Tie this to
    `model.py:51` reshaping `[:,120:512]` to `(E, 2*4, 49)`.
- `cache/liberty.pkl` — 6 MB pickle of the parsed Liberty (caps, arcs, checks,
  senses) so the 84 MB of `.lib` is parsed once.
- `checkpoints/` — `smoke1`, `smoke2`, `pathtest`: three ~0.9 MB runs from
  13 Oct. `smoke1/stdout.log` shows 60 epochs on a 7-design subset, R² still
  deeply negative at epoch 0 and losses falling (at 23.9 → 3.2). **The paper's
  pretrained `08_atcd_specul/15799.pth` is not present**, so `README.md`'s
  one-command reproduction cannot run as written.
- `.venv312/` — Python 3.12 venv, torch + `dgl 2.2.1` + numpy 1.26.4. DGL caps
  at ≤3.12, which is why this venv exists separately (pycache shows 3.14 was
  used for the stdlib-only scripts).

##### 6. Why so many Python files — grouped by author and role
Make the core point: the files split into **three generations**, and most of the
confusion is that generations 2 and 3 are not from the paper.

**(a) Upstream, used by the paper's headline result (4 files)**
`data_graph.py` (loads the 21 graphs, derives `n_net_delays_log`,
`n_atslew`, the topological level list, and the node-role index sets; fixes the
train/test split with `random.seed(8026728)` + `random.sample(..., 14)` →
14 train / 7 test), `model.py` (`MLP`, `NetConv`, `SignalProp`, `TimingGCN`),
`train_gnn.py` (train/test, `--test_iter`, `--checkpoint`), `tee.py` (log mirror).

**(b) Upstream baselines and side-experiments (6 files) — dead in this checkout**
`train_stat.py` + `data_stat.py` and `train_stat_rf.py` + `data_stat_cpu.py`
(MLP and random-forest net-delay baselines, need `data/4_netstat/`),
`train_deepgnn.py` + `data_homograph.py` (60-layer GCNII baseline, needs
`data/7_homotest/`), `plot_slacks.py` (needs `checkpoints/18_slacksdump/`).
None of those three directories is present, so these six files cannot run here —
they are why the file count looks inflated.

**(c) Added locally, 13–16 Oct 2025 (7 files + `openlane/`) — the real work**
This is the second, independent pipeline that rebuilds `data/8_rat` from
`netlists/` instead of downloading it:
| file | job |
|---|---|
| `device.py` | one place deciding CPU vs CUDA (`TIMINGPREDICT_DEVICE`); upstream hardcoded `.cuda()`, and DGL has no sm_120 build |
| `build_graph.py` | stdlib-only structural parse: LEF pin directions, Verilog instances/ports/aliases, SDF IOPATH/INTERCONNECT → the pin DAG. Also `--dir` mode and `parse_sdf_pair` for one-SDF-per-corner LibreLane output |
| `liberty_lut.py` | brace-depth Liberty parser → per-arc 7×7 LUTs, pin caps, timing senses, into `cache/liberty.pkl` |
| `sta_ref.py` | pure-Python levelized forward STA over the SDF-annotated DAG; produces the arrival-time labels |
| `make_dataset.py` | assembles everything into the DGL heterograph and writes `data/8_rat/*.graph.bin` |
| `openlane/` | `README.md`, `harden.sh`, `fix_sv.py`, `mkconfig.py`, `sta_dataset_style.tcl`, `tiny_gpu/config.ol{1,2}.json` + wrapper RTL — a path to harden [adam-maj/tiny-gpu](https://github.com/adam-maj/tiny-gpu) in sky130 as a 22nd, GPU design |

Also document the local changes to upstream (`git diff`: 4 files, +56/−19) —
`.cuda()` → `device.py`, int32 → int64 node ids for DGL ≥ 2,
`random.sample(dict.items())` → `list(...)` for Python ≥ 3.11, and the
`TIMINGPREDICT_DESIGNS` / `_EXTRA` / `_BATCH` / `_EPOCHS` env knobs that make a
21-graph, several-GB dataset trainable on a 16 GB laptop.

##### 7. The model and the task
Short, concrete: `TimingGCN` = 3 stacked `NetConv` layers (10→32→32→16; the
first 4 of those 16 outputs *are* the net-delay prediction) feeding a
`SignalProp` that walks `ts['topo']` level by level, alternating net-arc
propagation and cell-arc propagation — a learned mimic of an STA engine's
levelized forward pass, which is the paper's central idea. Explain the
differentiable LUT lookup in `SignalProp.edge_msg_cell` (MLP emits an axis
query → attention over `index_1`/`index_2` → outer product → batched dot with
the flattened 7×7 table), and the `groundtruth=True` teacher-forcing mode. Loss
= MSE on net delay (log-scaled) + cell delay + `n_atslew`. Mention the known
`r2_score(pred, truth)` argument-order typo the authors deliberately keep
(issue #7) so numbers match the paper.

##### 8. Verified status and the gaps that matter
Report plainly what was checked in this session:
- **Structure reproduces exactly.** `build_graph.py netlists spm usb
  cic_decimator zipdiv xtea wbqspiflash usb_cdc_core` → `EXACT` on all four
  counts for all 7. The "unmapped net" counts it prints are the VGND/VPWR
  `INTERCONNECT` records into filler cells, correctly dropped.
- **The labels are not the authors'.** `data/8_rat` here was generated locally
  (mtimes 13 Oct 18:42–18:43, right after `cache/liberty.pkl` at 18:39), with
  labels from `sta_ref.py`, not from the `.at.sdf` `(ARRIVALTIMES)` block.
  Two concrete consequences, both verified on spm:
  1. `n_slews` is **all zero** (0 of 4,484 entries non-zero) — yet `.at.sdf`
     carries real `(SLEW ...)` per pin. Half of `n_atslew`'s 8 regression
     channels are therefore a constant, and `log(0.0001+0)+3 ≈ −6.21`.
  2. `n_ats` is cone-relative (max 2.77 ns for spm) because `sta_ref.py` starts
     every primary input and every flop output at 0, ignoring SDC input delays
     and clock latency — whereas `.at.sdf` has absolute STA arrivals
     (`AT rst = 2.0`, matching `set_input_delay 2.0`; `AT clk = 0::0 / 5::5`).
  So this checkout is self-consistent (train and score here and it works) but
  not numerically the published dataset — which also explains why
  `openlane/README.md` warns the pretrained checkpoint "is not expected to
  transfer".
- **The cheap fix to flag:** parse `(AT ...)` / `(SLEW ...)` / `(RAT ...)` out of
  the `.at.sdf` that is already on disk. `build_graph.py:parse_sdf` currently
  reads only `IOPATH` and `INTERCONNECT`. That recovers real slews, absolute
  arrivals, and — via `RAT − AT` — the slack the paper's title is about.
- The GPU extension is written but **not yet run**: `openlane/hardened/` does not
  exist, and the flow needs Linux/WSL2/Colab (no OpenROAD/OpenSTA on Windows).
- `data/4_netstat`, `data/7_homotest`, `checkpoints/18_slacksdump`,
  `checkpoints/08_atcd_specul` are all absent.

##### 9. Quick reference appendix
- Disk map: `netlists/` 3.5 GB raw · `data/` 2.2 GB graphs · `techlib/` 83 MB ·
  `cache/` 6 MB · `checkpoints/` 2.7 MB · `openlane/` 48 KB.
- The commands that actually work in this checkout, in order (liberty cache →
  `build_graph.py` sanity check → `make_dataset.py` → `train_gnn.py` with the
  `TIMINGPREDICT_*` env vars), each with the `.venv312/Scripts/python.exe` prefix.
- `.gitignore` reading: `netlists/`, `data/`, `cache/`, `checkpoints/` and the
  OpenLane run artifacts are all untracked — so only the ~17 `.py` files and
  `openlane/` are version-controlled, which is the other half of the answer to
  "why so many Python files".

#### Writing constraints
- Facts only from what was verified above; no invented R² numbers, no claims
  about the paper's results beyond what `README.md` and the code state.
- Mark the §8 label findings clearly as findings about *this checkout*, not about
  the published dataset.
- Use they/them for the developer; do not name them.
- Do not modify any file other than `my_work/analysis.md`.

#### Verification
1. `my_work/analysis.md` is non-empty and renders (headings, tables well-formed).
2. Spot-check every number in it against this session's outputs: the REF table,
   the clock/COMPONENTS/DIEAREA table, the spm tensor shapes, the
   5,354/11,473-line SDF comparison, the 7× `EXACT` run, 441 LEF MACROs,
   `SITE 0.46×2.72`, `dgl 2.2.1`.
3. Confirm no other file in the repo changed: `git status` shows only the new
   `my_work/analysis.md` beyond the pre-existing modifications.
