# Mental Health EDA Summary

## Dataset
- Source: NHANES 2017–2018
- Participants: 5533
- Columns: 14
- Duplicate participant IDs: 0

## PHQ-9 category counts

phq9_category
minimal              3772
mild                  837
NaN                   465
moderate              292
moderately severe     124
severe                 43

## Missing values

seqn               0
phq9_score       465
phq9_category    465
height_cm         89
weight_kg         89
bmi               99
waist_cm         349
sbp              283
dbp              297
pulse            264
told_high_bp      10
total_chol       357
hdl              357
hba1c            272

## Clinical feature summary

           bmi      sbp      dbp    pulse  total_chol      hdl    hba1c
count  5434.00  5250.00  5236.00  5269.00     5176.00  5176.00  5261.00
mean     29.69   126.35    72.48    71.88      186.88    53.19     5.84
std       7.44    20.12    11.99    11.74       41.24    15.55     1.09
min      14.20    72.70     8.00    34.00       76.00    10.00     3.80
1%       17.53    94.00    44.91    50.00      107.00    28.00     4.60
25%      24.60   112.00    64.70    64.00      158.00    42.00     5.30
50%      28.50   123.30    72.70    70.00      183.00    51.00     5.60
75%      33.50   136.70    80.00    78.00      212.00    61.00     6.00
99%      53.43   188.70   104.00   104.00      300.00    97.00    10.64
max      86.20   238.00   135.30   136.00      446.00   189.00    16.20

## IQR outlier counts (investigation flags, not automatic errors)

- bmi: below=0, above=156; bounds=(11.25, 46.85)
- sbp: below=2, above=149; bounds=(74.95, 173.75)
- dbp: below=28, above=57; bounds=(41.75, 102.95)
- pulse: below=4, above=103; bounds=(43.00, 99.00)
- total_chol: below=1, above=68; bounds=(77.00, 293.00)
- hdl: below=2, above=126; bounds=(13.50, 89.50)
- hba1c: below=10, above=455; bounds=(4.25, 7.05)

## Cleaning notes
- IQR flags are statistical indicators, not proof of invalid measurements.
- Extreme values have not been automatically removed. Validate against source records before changing them.
- PHQ-9 categories are derived from PHQ-9 scores; do not use the score as a predictor of its derived category.
- This report is descriptive and does not establish causation.
