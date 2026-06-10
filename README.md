# PropKot

Cycle analysis tools for a turbocharger-core turbojet engine using 
the BorgWarner S400SX-E as the compressor-turbine core.

## Project Overview

PropKot is a low-cost expendable turbojet engine project, currently in 
Phase 1 (theoretical analysis). The full project documentation is 
available in the latest project PDF.

## Repository Contents

- `properties.py` — Thermodynamic property tables and interpolation 
  functions (based on Mattingly Appendix D)
- `stations.py` — Component analysis functions (compressor, combustor, 
  turbine, nozzle)
- `analyze.py` — Cycle analysis driver, parametric sweeps, plotting

## Quick Start

```python
from analyze import run_cycle, print_summary

result = run_cycle(pi_c=3.4, eta_c=0.72, mdot_air=0.91, T_t4=1250)
print_summary(result)
```

## License

Code released under the MIT License (see LICENSE).
Documentation released under CC BY 4.0.

## Author

Andrii Kotenev, 2026
