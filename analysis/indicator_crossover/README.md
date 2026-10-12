# Indicator crossover (Sec. 6.1)

`indicator_crossover.cpp` evaluates HULK's indicator [S == n] for a sum S in 0..n with either the
root-paired factorization or Paterson-Stockmeyer and checks every slot.

```bash
g++ -std=c++20 -O3 -I../../common -I$SEAL_DIR/native/src -I$SEAL_DIR/build/native/src \
    -I$SEAL_DIR/build/thirdparty/msgsl-src/include indicator_crossover.cpp -o indicator_crossover \
    -L$SEAL_DIR/build/lib -lseal-4.1 -lpthread
./indicator_crossover <n> <limbs> <method: 0 = root-paired, 1 = Paterson-Stockmeyer>
```
