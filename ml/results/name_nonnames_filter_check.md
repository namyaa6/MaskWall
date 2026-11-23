# Non-name filter check (Day 4.24)

Same raw model output scored without (before) and with (after) `dropNonNames`. Strict recall = every character of a gold name covered.

| Mode | Set (sentences) | Names found before | after | False-positive spans before | after | on harmless sentences before | after |
|---|---|---|---|---|---|---|---|
| A (standard mode) | existing (62) | 47/56 | 47/56 | 5 | 5 | - | - |
| A (standard mode) | dev (120) | 11/66 | 11/66 | 1 | 1 | - | - |
| A (standard mode) | fresh (120) | 10/60 | 10/60 | 0 | 0 | - | - |
| B2 (strict mode) | existing (62) | 53/56 | 53/56 | 11 | 11 | - | - |
| B2 (strict mode) | dev (120) | 49/66 | 49/66 | 20 | 20 | 13 | 13 |
| B2 (strict mode) | fresh (120) | 48/60 | 48/60 | 28 | 28 | 21 | 21 |

**Name recall did not drop in any set or mode.**

Spans removed by the filter across all sets: 6.
- A (standard), custom: removed "Aadhaar" in "Aadhaar 1234 5768 9881 and PAN ABCDE1234F"
- A (standard), custom: removed "Aadhaar" in "Rahul Sharma's Aadhaar 1234 5768 9881 and PAN ABCDE1234F"
- A (standard), custom: removed "aadhaar" in "aadhaar 1234 5768 9881 and pan abcde1234f"
- B2 (strict), custom: removed "Aadhaar" in "Aadhaar 1234 5768 9881 and PAN ABCDE1234F"
- B2 (strict), custom: removed "Aadhaar" in "Rahul Sharma's Aadhaar 1234 5768 9881 and PAN ABCDE1234F"
- B2 (strict), custom: removed "aadhaar" in "aadhaar 1234 5768 9881 and pan abcde1234f"

## Sentences with ID-document terms (no gold names; model output before -> after)
- A (standard): "Aadhaar 1234 5768 9881 and PAN ABCDE1234F": before ["Aadhaar"], after []
- A (standard): "My Aadhaar is 1234 5768 9881 and my PAN is ABCDE1234F.": before [], after []
- A (standard): "Rahul Sharma's Aadhaar 1234 5768 9881 and PAN ABCDE1234F": before ["Rahul Sharma","Aadhaar"], after ["Rahul Sharma"]
- A (standard): "Please verify my Passport and Voter ID, my UPI id and the OTP for KYC.": before [], after []
- A (standard): "GSTIN 27ABCDE1234F1Z5 and IFSC HDFC0001234 are on the invoice": before [], after []
- A (standard): "aadhaar 1234 5768 9881 and pan abcde1234f": before ["aadhaar"], after []
- B2 (strict): "Aadhaar 1234 5768 9881 and PAN ABCDE1234F": before ["Aadhaar"], after []
- B2 (strict): "My Aadhaar is 1234 5768 9881 and my PAN is ABCDE1234F.": before [], after []
- B2 (strict): "Rahul Sharma's Aadhaar 1234 5768 9881 and PAN ABCDE1234F": before ["Rahul Sharma","Aadhaar"], after ["Rahul Sharma"]
- B2 (strict): "Please verify my Passport and Voter ID, my UPI id and the OTP for KYC.": before [], after []
- B2 (strict): "GSTIN 27ABCDE1234F1Z5 and IFSC HDFC0001234 are on the invoice": before [], after []
- B2 (strict): "aadhaar 1234 5768 9881 and pan abcde1234f": before ["aadhaar"], after []
