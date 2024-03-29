## Environment

- install openCV library
- install cuda library (NVCC)
- install openMP library

## How to use

run pthread test:
```
make pthread_test
```

run omp test:
```
make omp_test
```

run cuda test:
```
make cuda_test
```

### delete all test files
```
make clean
```

## Excepcted result

pthread test:
```
--------------------------------------------------------------------------
|                            Pthread test                                |
--------------------------------------------------------------------------
| Thread num || Total time | Eff. || DFT time | Eff. || IDFT time | Eff. |
--------------------------------------------------------------------------
| Serial     || 15.8787    | N/A  || 11.8507  | N/A  || 4.00281   | N/A  |
| 2          || 8.17436    | 0.97 || 6.08283  | 0.97 || 2.0668    | 0.97 |
| 4          || 4.20027    | 0.95 || 3.11125  | 0.95 || 1.06684   | 0.94 |
| 8          || 2.64592    | 0.75 || 1.9134   | 0.77 || 0.708024  | 0.71 |
| 16         || 2.3554     | 0.42 || 1.79136  | 0.41 || 0.540491  | 0.46 |
--------------------------------------------------------------------------
```

omp test:
```
--------------------------------------------------------------------------
|                             OpenMP test                                |
--------------------------------------------------------------------------
| Thread num || Total time | Eff. || DFT time | Eff. || IDFT time | Eff. |
--------------------------------------------------------------------------
| Serial     || 16.0617    | N/A  || 12.0432  | N/A  || 3.99476   | N/A  |
| 2          || 8.21131    | 0.98 || 6.12962  | 0.98 || 2.05526   | 0.97 |
| 4          || 4.18528    | 0.96 || 3.12526  | 0.96 || 1.0335    | 0.97 |
| 8          || 2.46982    | 0.81 || 1.81825  | 0.83 || 0.625604  | 0.80 |
| 16         || 1.60615    | 0.63 || 1.18217  | 0.64 || 0.398181  | 0.63 |
--------------------------------------------------------------------------
```

cuda test:
```
--------------------------------------------------------------------------------------
|                                    CUDA test                                       |
--------------------------------------------------------------------------------------
| Thread num || Total time | SpeedUp. || DFT time | SpeedUp. || IDFT time | SpeedUp. |
--------------------------------------------------------------------------------------
| Serial     || 16.1653    | N/A      || 12.0682  | N/A      || 4.0695    | N/A      |
| CUDA       || 0.970525   | 16.66    || 0.762929 | 15.82    || 0.182541  | 22.29    |
--------------------------------------------------------------------------------------
```