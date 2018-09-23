# Natural Memory Layout #

## Naive CPU ##
  N                  = 10  
  Min                = 2258.59 ms  
  Mean               = 2320.94 ms  
  Standard Deviation = 35.5085 ms  

## Naive CPU with /arch:AVX2 ##
  N                  = 10  
  Min                = 2361.67 ms  
  Mean               = 2428.31 ms  
  Standard Deviation = 52.1358 ms  

## Naive CPU with /arch:AVX2 and OpenMP ##
  N                  = 10  
  Min                = 2141.51 ms  
  Mean               = 2400.09 ms  
  Standard Deviation = 137.464 ms  
  (why no speedup?)

## AMP ##
  N                  = 10  
  Min                = 56.0259 ms  
  Mean               = 60.7515 ms  
  Standard Deviation = 4.48584 ms  

# Opimized Memory Layout (transposed B) #
## Naive CPU ##
  N                  = 10
  Min                = 1467.04
  Mean               = 1496.86
  Standard Deviation = 15.4569

## Naive CPU with /arch:AVX2 ##
  N                  = 10
  Min                = 1469.04
  Mean               = 1539.99
  Standard Deviation = 59.0066

## Naive CPU with /arch:AVX2 and OpenMP ##
  N                  = 10
  Min                = 200.127
  Mean               = 242.77
  Standard Deviation = 22.7793

## AMP ##
  N                  = 10
  Min                = 343.243
  Mean               = 355.658
  Standard Deviation = 9.37897
  (gpu why?)