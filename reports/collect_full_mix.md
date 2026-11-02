# Collection mix for review

150 agreements picked, one per issuer, from 1250 exhibits examined. The archetype comes from defined terms (`src/discover.py`, `classify`) and needs your review: unsecured is investment grade, and secured with two or more leveraged-loan terms is leveraged.

| archetype | quota | picked | train | test |
|---|---|---|---|---|
| investment_grade | 30 | 30 | 24 | 6 |
| secured_mid_cap | 60 | 60 | 48 | 12 |
| leveraged | 60 | 60 | 48 | 12 |

## Exhibits not picked

| reason | exhibits |
|---|---|
| an amendment | 318 |
| quota full | 302 |
| title is not a credit agreement: '' | 233 |
| N charactersN outside N to N | 55 |
| HTML size N outside N to N | 44 |
| a fund or BDC facility | 26 |
| agreement dated unknownN before N-N-N | 13 |
| title is not a credit agreement: 'LOAN AND SECURITY AGREEMENT' | 10 |
| title is not a credit agreement: 'RECEIVABLES FINANCING AGREEMENT' | 5 |
| title is not a credit agreement: 'RECEIVABLES PURCHASE AGREEMENT' | 4 |
| title is not a credit agreement: 'AGREEMENT' | 4 |
| agreement dated N-N-N before N-N-N | 4 |
| title is not a credit agreement: 'LOAN FINANCING AND SERVICING AGREEME | 4 |
| title is not a credit agreement: 'AMENDMENT AND RESTATEMENT AGREEMENT' | 4 |
| title is not a credit agreement: 'FINANCING AGREEMENT' | 3 |
| title is not a credit agreement: 'AMENDMENT NO. N TO RECEIVABLES PURCH | 2 |
| title is not a credit agreement: 'BRIDGE LOAN AND SECURITY AGREEMENT' | 2 |
| title is not a credit agreement: 'TO SYNDICATED FACILITY AGREEMENT' | 2 |
| title is not a credit agreement: 'LOAN AND SERVICING AGREEMENT' | 2 |
| title is not a credit agreement: 'TO LOAN AND SECURITY AGREEMENT' | 2 |
| title is not a credit agreement: 'AMENDMENT NO. N TO LOAN AND SECURITY | 2 |
| title is not a credit agreement: 'RECEIVABLES PURCHASE AND FINANCING A | 2 |
| title is not a credit agreement: 'CONFORMED THROUGH AMENDMENT #N TO RE | 2 |
| title is not a credit agreement: 'WAREHOUSE AGREEMENT' | 2 |
| title is not a credit agreement: 'SECTION N. GOVERNING LAW. THIS AGREE | 1 |
| title is not a credit agreement: 'AMENDMENT NO. N TO MASTER ACCOUNTS R | 1 |
| title is not a credit agreement: 'SECTION N. AMENDMENTS TO LOAN AND SE | 1 |
| title is not a credit agreement: 'FIRST AMENDMENT TO FINANCING AGREEME | 1 |
| title is not a credit agreement: 'TERM LOANN GUARANTEE AND SECURITY AG | 1 |
| title is not a credit agreement: 'MORTGAGE WAREHOUSING AGREEMENT' | 1 |
| title is not a credit agreement: 'RECEIVABLES FACILITY LOAN AND SECURI | 1 |
| title is not a credit agreement: 'MULTICURRENCY TERM AND REVOLVING FAC | 1 |
| title is not a credit agreement: 'SYNDICATED FACILITY AGREEMENT' | 1 |
| title is not a credit agreement: 'UNDER THIS AGREEMENT' | 1 |
| title is not a credit agreement: 'SIXTH MODIFICATION AGREEMENT' | 1 |
| title is not a credit agreement: 'FIFTH MODIFICATION AGREEMENT' | 1 |
| title is not a credit agreement: 'LOANN GUARANTY AND SECURITY AGREEMEN | 1 |
| title is not a credit agreement: 'TERM LOAN EXCHANGE AGREEMENT' | 1 |
| title is not a credit agreement: 'DEBTOR-IN-POSSESSION FINANCING AGREE | 1 |
| title is not a credit agreement: 'TO FINANCING AGREEMENT' | 1 |
| title is not a credit agreement: 'FIFTH AMENDMENT TO LOAN AND SECURITY | 1 |
| title is not a credit agreement: 'THIRD AMENDMENT TO LOAN AND SECURITY | 1 |
| title is not a credit agreement: 'THIRD AMENDMENT AND RESTATEMENT AGRE | 1 |
| title is not a credit agreement: 'N. CONDITIONS; TERM OF AGREEMENT N' | 1 |
| title is not a credit agreement: 'AMENDMENT NO. N TO FINANCING AGREEME | 1 |
| title is not a credit agreement: 'TRANSACTION AGREEMENT' | 1 |
| title is not a credit agreement: 'AGREEMENTS' | 1 |
| title is not a credit agreement: 'RESTATEMENT AGREEMENT' | 1 |
| title is not a credit agreement: 'REVOLVING FACILITY REFINANCING AGREE | 1 |
| title is not a credit agreement: 'FOURTH AMENDMENT TO LOAN AND SECURIT | 1 |
| title is not a credit agreement: 'SENIOR NOTES PURCHASE AGREEMENT' | 1 |
| title is not a credit agreement: 'ANNEX A TO SECONDTHIRD AMENDMENT AGR | 1 |
| title is not a credit agreement: 'N REPLACEMENT FACILITY AMENDMENT AND | 1 |
| title is not a credit agreement: 'RESTRUCTURING SUPPORT AGREEMENT' | 1 |
| title is not a credit agreement: 'NOTE PURCHASE AGREEMENT' | 1 |
| title is not a credit agreement: 'SIXTH AMENDMENT TO AMENDED AND RESTA | 1 |
| title is not a credit agreement: 'TO FOURTH AMENDMENT AGREEMENT' | 1 |
| title is not a credit agreement: 'TO THIRD AMENDMENT AGREEMENT' | 1 |
| title is not a credit agreement: 'LIMITED WAIVER AND FIFTH AMENDMENT T | 1 |
| title is not a credit agreement: 'SECOND AMENDMENT TO FINANCING AGREEM | 1 |
| title is not a credit agreement: 'SECOND JOINDER AND RESTATEMENT AGREE | 1 |
| title is not a credit agreement: 'SECOND AMENDMENT TO LOAN FACILITY AG | 1 |
| title is not a credit agreement: 'THIRD AMENDED AND RESTATED LOAN AND  | 1 |
| title is not a credit agreement: 'FOURTH AMENDED AND RESTATED SYNDICAT | 1 |
| title is not a credit agreement: 'CERTAIN INFORMATION HAS BEEN EXCLUDE | 1 |
| title is not a credit agreement: 'FIRST AMENDMENT TO LOAN AND SERVICIN | 1 |
| title is not a credit agreement: 'AND CONTINUING COVENANT AGREEMENT' | 1 |
| title is not a credit agreement: 'SECOND AMENDMENT AND RESTATEMENT AGR | 1 |
| title is not a credit agreement: 'SUCCESSOR AGENT APPOINTMENT AND AGEN | 1 |
| title is not a credit agreement: 'THIRD AMENDED AND RESTATED SYNDICATE | 1 |
| title is not a credit agreement: 'SECOND AMENDED AND RESTATED LOAN AND | 1 |
| title is not a credit agreement: 'AMENDED AND RESTATED LOAN AND SECURI | 1 |
| title is not a credit agreement: 'SECTION N.N. COMPLIANCE WITH LAWS AN | 1 |
| title is not a credit agreement: 'FIFTH AMENDMENT AND RESTATEMENT AGRE | 1 |

## Picks

| id | issuer | agreement date | characters | archetype | split | secured terms | leveraged terms | rating terms |
|---|---|---|---|---|---|---|---|---|
| [apache-6769](https://www.sec.gov/Archives/edgar/data/6769/000119312524018915/d760841dex102.htm) | APACHE CORP  (CIK 0000006769) | 2024-01-30 | 292,776 | investment_grade | train | 0 |  | 1 |
| [asbury-1144980](https://www.sec.gov/Archives/edgar/data/1144980/000114498025000149/ex101termloanagreement-asb.htm) | ASBURY AUTOMOTIVE GROUP INC  (ABG)  (CIK | 2025-07-21 | 503,269 | investment_grade | train | 0 |  | 0 |
| [at-732717](https://www.sec.gov/Archives/edgar/data/732717/000119312525262515/d935272dex101.htm) | AT&T INC.  (T, TBB, T-PA, T-PC)  (CIK 00 | 2025-11-03 | 321,545 | investment_grade | train | 0 |  | 1 |
| [cabot-16040](https://www.sec.gov/Archives/edgar/data/16040/000119312526222854/d31752dex101.htm) | CABOT CORP  (CBT)  (CIK 0000016040) | 2026-05-12 | 462,486 | investment_grade | test | 0 |  | 1 |
| [campbell-16732](https://www.sec.gov/Archives/edgar/data/16732/000119312524097732/d806817dex10.htm) | CAMPBELL SOUP CO  (CPB)  (CIK 0000016732 | 2024-04-16 | 348,174 | investment_grade | train | 0 |  | 0 |
| [century-1576940](https://www.sec.gov/Archives/edgar/data/1576940/000114036124045029/ef20037977_ex10-1.htm) | Century Communities, Inc.  (CCS)  (CIK 0 | 2024-11-01 | 414,015 | investment_grade | train | 0 |  | 0 |
| [cerus-1020214](https://www.sec.gov/Archives/edgar/data/1020214/000119312526326295/cers-ex10_1.htm) | CERUS CORP  (CERS)  (CIK 0001020214) | 2026-06-05 | 489,978 | investment_grade | train | 0 |  | 0 |
| [cf-1324404](https://www.sec.gov/Archives/edgar/data/1324404/000110465925088627/tm2525195d1_ex10-1.htm) | CF Industries Holdings, Inc.  (CF)  (CIK | 2025-09-04 | 506,634 | investment_grade | train | 0 |  | 1 |
| [cnl-1496454](https://www.sec.gov/Archives/edgar/data/1496454/000119312523293332/d652210dex101.htm) | CNL Healthcare Properties, Inc.  (CHTH)  | 2023-12-07 | 378,790 | investment_grade | train | 0 |  | 0 |
| [csx-277948](https://www.sec.gov/Archives/edgar/data/277948/000095010323003669/dp190172_ex1001.htm) | CSX CORP  (CSX)  (CIK 0000277948) | 2023-02-28 | 347,662 | investment_grade | train | 0 |  | 1 |
| [edison-827052](https://www.sec.gov/Archives/edgar/data/827052/000082705225000119/eix-20251223xex10d1.htm) | EDISON INTERNATIONAL  (EIX)  (CIK 000082 | 2025-12-23 | 201,797 | investment_grade | train | 0 |  | 0 |
| [ge-1996810](https://www.sec.gov/Archives/edgar/data/1996810/000199681024000008/gev1q2024exhibit101.htm) | GE Vernova Inc.  (GEV)  (CIK 0001996810) | 2024-03-26 | 356,966 | investment_grade | train | 0 |  | 1 |
| [gxo-1852244](https://www.sec.gov/Archives/edgar/data/1852244/000110465924041819/tm2410290d1_ex10-1.htm) | GXO Logistics, Inc.  (GXO)  (CIK 0001852 | 2024-03-29 | 334,398 | investment_grade | train | 0 |  | 0 |
| [honeywell-773840](https://www.sec.gov/Archives/edgar/data/773840/000119312526095179/d71810dex101.htm) | HONEYWELL INTERNATIONAL INC  (HON)  (CIK | 2026-03-06 | 259,350 | investment_grade | train | 0 |  | 1 |
| [hp-47217](https://www.sec.gov/Archives/edgar/data/47217/000004721724000063/ex101-hpx5xyearrevolvercre.htm) | HP INC  (HPQ)  (CIK 0000047217) | 2024-08-01 | 382,822 | investment_grade | train | 0 |  | 2 |
| [keurig-1418135](https://www.sec.gov/Archives/edgar/data/1418135/000141813525000051/kdp-creditagreement2025.htm) | Keurig Dr Pepper Inc.  (KDP)  (CIK 00014 | 2025-03-31 | 377,682 | investment_grade | test | 0 |  | 1 |
| [lam-707549](https://www.sec.gov/Archives/edgar/data/707549/000070754925000011/lamresearch-thirdarcredita.htm) | LAM RESEARCH CORP  (LRCX)  (CIK 00007075 | 2025-01-27 | 534,168 | investment_grade | train | 0 |  | 1 |
| [landsea-1721386](https://www.sec.gov/Archives/edgar/data/1721386/000173112224000689/e5620_ex10-1.htm) | Landsea Homes Corp  (LSEA, LSEAW)  (CIK  | 2024-04-19 | 425,669 | investment_grade | train | 0 |  | 0 |
| [lazard-1326141](https://www.sec.gov/Archives/edgar/data/1326141/000162828023026251/group-20230630xex1021.htm) | Lazard Group LLC  (CIK 0001326141) | 2023-06-06 | 262,715 | investment_grade | test | 0 |  | 1 |
| [lockheed-936468](https://www.sec.gov/Archives/edgar/data/936468/000119312526371750/d177255dex101.htm) | LOCKHEED MARTIN CORP  (LMT)  (CIK 000093 | 2026-08-24 | 234,074 | investment_grade | train | 0 |  | 0 |
| [marketaxess-1278021](https://www.sec.gov/Archives/edgar/data/1278021/000119312523207377/d529217dex101.htm) | MARKETAXESS HOLDINGS INC  (MKTX)  (CIK 0 | 2023-08-09 | 413,225 | investment_grade | train | 0 |  | 1 |
| [ovintiv-1792580](https://www.sec.gov/Archives/edgar/data/1792580/000119312525304036/d62184dex101.htm) | Ovintiv Inc.  (OVV)  (CIK 0001792580) | 2025-11-25 | 395,740 | investment_grade | train | 0 |  | 1 |
| [patterson-889900](https://www.sec.gov/Archives/edgar/data/889900/000088990025000028/ex101pten-rcfcreditagreeme.htm) | PATTERSON UTI ENERGY INC  (PTEN)  (CIK 0 | 2025-01-31 | 415,796 | investment_grade | train | 0 |  | 1 |
| [plains-1070423](https://www.sec.gov/Archives/edgar/data/1070423/000110465926075186/tm2618132d1_ex10-1.htm) | PLAINS ALL AMERICAN PIPELINE LP  (PAA, P | 2026-06-12 | 488,732 | investment_grade | train | 0 |  | 2 |
| [rubicon-1862068](https://www.sec.gov/Archives/edgar/data/1862068/000182912623003991/rubicontech_ex10-2.htm) | Rubicon Technologies, Inc.  (RBTC, RBTCW | 2023-06-07 | 470,157 | investment_grade | test | 0 |  | 0 |
| [salesforce-1108524](https://www.sec.gov/Archives/edgar/data/1108524/000119312526104282/d887348dex102.htm) | Salesforce, Inc.  (CRM)  (CIK 0001108524 | 2026-03-11 | 309,439 | investment_grade | train | 0 |  | 1 |
| [servicenow-1373715](https://www.sec.gov/Archives/edgar/data/1373715/000137371526000076/a103-termloancreditagr.htm) | ServiceNow, Inc.  (NOW)  (CIK 0001373715 | 2026-04-17 | 324,325 | investment_grade | test | 0 |  | 1 |
| [steris-1757898](https://www.sec.gov/Archives/edgar/data/1757898/000119312524233855/d833057dex101.htm) | STERIS plc  (STE)  (CIK 0001757898) | 2024-10-07 | 458,391 | investment_grade | test | 0 |  | 2 |
| [tpg-1880661](https://www.sec.gov/Archives/edgar/data/1880661/000188066125000027/seventhamendedandrestatedc.htm) | TPG Inc.  (TPG, TPGXL)  (CIK 0001880661) | 2025-05-01 | 430,340 | investment_grade | train | 0 |  | 0 |
| [western-1423902](https://www.sec.gov/Archives/edgar/data/1423902/000142390223000034/wes20238-kxaprilrcfxex101.htm) | Western Midstream Partners, LP  (WES)  ( | 2023-04-06 | 372,090 | investment_grade | train | 0 |  | 1 |
| [advansix-1673985](https://www.sec.gov/Archives/edgar/data/1673985/000095015726000916/ex10-1.htm) | AdvanSix Inc.  (ASIX)  (CIK 0001673985) | 2026-08-14 | 617,887 | leveraged | train | 3 | Available Amount, Incremental Equivalent Debt | 0 |
| [agiliti-1749704](https://www.sec.gov/Archives/edgar/data/1749704/000174970423000024/agti-ex101xcreditagreement.htm) | AGILITI, INC. \DE  (CIK 0001749704) | 2023-05-01 | 823,125 | leveraged | train | 1 | Available Amount, Incremental Equivalent Debt, Excess Cash Flow | 0 |
| [akumin-1776197](https://www.sec.gov/Archives/edgar/data/1776197/000119312524025866/d31963dex103.htm) | AKUMIN INC.  (CIK 0001776197) | 2024-02-06 | 626,390 | leveraged | train | 3 | Cumulative Credit, Incremental Equivalent Debt | 0 |
| [alkermes-1520262](https://www.sec.gov/Archives/edgar/data/1520262/000119312526047682/d43532dex101.htm) | Alkermes plc.  (ALKS)  (CIK 0001520262) | 2026-02-12 | 726,350 | leveraged | train | 1 | Available Amount, Incremental Equivalent Debt, Excess Cash Flow, Effective Yield | 0 |
| [alnylam-1178670](https://www.sec.gov/Archives/edgar/data/1178670/000119312525230201/d19737dex101.htm) | ALNYLAM PHARMACEUTICALS, INC.  (ALNY)  ( | 2025-09-30 | 700,397 | leveraged | test | 3 | Incremental Equivalent Debt, Ratio Debt | 1 |
| [amc-1411579](https://www.sec.gov/Archives/edgar/data/1411579/000110465926113853/tm2627016d1_ex4-3.htm) | AMC ENTERTAINMENT HOLDINGS, INC.  (AMC)  | 2026-10-05 | 669,354 | leveraged | train | 3 | Excess Cash Flow, Effective Yield | 0 |
| [ani-1023024](https://www.sec.gov/Archives/edgar/data/1023024/000095010324012144/dp216536_ex1002.htm) | ANI PHARMACEUTICALS INC  (ANIP)  (CIK 00 | 2024-08-13 | 694,083 | leveraged | train | 2 | Available Amount, Incremental Equivalent Debt | 0 |
| [astrana-1083446](https://www.sec.gov/Archives/edgar/data/1083446/000110465925018221/tm257416d1_ex10-1.htm) | Astrana Health, Inc.  (ASTH)  (CIK 00010 | 2025-02-26 | 664,104 | leveraged | train | 1 | Available Amount, Incremental Equivalent Debt | 0 |
| [audacy-1067837](https://www.sec.gov/Archives/edgar/data/1067837/000119312524229306/d753507dex101.htm) | AUDACY, INC.  (AUDAQ)  (CIK 0001067837) | 2024-09-30 | 520,799 | leveraged | train | 3 | Excess Cash Flow, All-In Yield | 1 |
| [berry-1378992](https://www.sec.gov/Archives/edgar/data/1378992/000114036123038801/exhibit10_1.htm) | BERRY GLOBAL GROUP, INC.  (BERY)  (CIK 0 | 2023-06-22 | 887,895 | leveraged | train | 2 | Cumulative Credit, Excess Cash Flow | 0 |
| [biomarin-1048477](https://www.sec.gov/Archives/edgar/data/1048477/000119312526179243/d113922dex101.htm) | BIOMARIN PHARMACEUTICAL INC  (BMRN)  (CI | 2026-04-27 | 890,845 | leveraged | train | 2 | Available Amount, Excess Cash Flow, Term B Loans | 0 |
| [boyd-906553](https://www.sec.gov/Archives/edgar/data/906553/000119312526017988/d95524dex101.htm) | BOYD GAMING CORP  (BYD)  (CIK 0000906553 | 2026-01-21 | 899,847 | leveraged | train | 3 | Available Amount, Excess Cash Flow, Ratio Debt | 0 |
| [bridger-1941536](https://www.sec.gov/Archives/edgar/data/1941536/000168316825007822/bridger_ex1001.htm) | Bridger Aerospace Group Holdings, Inc.   | 2025-10-28 | 575,282 | leveraged | train | 2 | Available Amount, Excess Cash Flow, Effective Yield | 0 |
| [brinks-78890](https://www.sec.gov/Archives/edgar/data/78890/000007889026000021/exhibit1012.htm) | BRINKS CO  (BCO)  (CIK 0000078890) | 2026-03-31 | 920,314 | leveraged | train | 2 | Incremental Equivalent Debt, All-In Yield | 0 |
| [caci-16058](https://www.sec.gov/Archives/edgar/data/16058/000162828025054481/secondar1.htm) | CACI INTERNATIONAL INC /DE/  (CACI)  (CI | 2025-11-25 | 568,225 | leveraged | test | 2 | All-In Yield, Ratio Debt | 0 |
| [cano-1800682](https://www.sec.gov/Archives/edgar/data/1800682/000119312524172745/d840809dex102.htm) | Cano Health, Inc.  (CIK 0001800682) | 2024-06-28 | 841,761 | leveraged | train | 2 | Available Amount, Incremental Equivalent Debt, Excess Cash Flow, Effective Yield, Ratio Debt | 0 |
| [chefs-1517175](https://www.sec.gov/Archives/edgar/data/1517175/000151717526000036/chefs-termloancreditagreem.htm) | Chefs' Warehouse, Inc.  (CHEF)  (CIK 000 | 2026-10-15 | 503,233 | leveraged | test | 3 | Available Amount, Incremental Equivalent Debt, Excess Cash Flow | 0 |
| [clearwater-1866368](https://www.sec.gov/Archives/edgar/data/1866368/000119312525087277/d926858dex101.htm) | Clearwater Analytics Holdings, Inc.  (CW | 2025-04-21 | 790,014 | leveraged | train | 3 | Available Amount, Excess Cash Flow | 0 |
| [contextlogic-2064307](https://www.sec.gov/Archives/edgar/data/2064307/000119312526094012/logc-ex10_26.htm) | ContextLogic Holdings Inc.  (LOGC)  (CIK | 2026-02-26 | 966,604 | leveraged | train | 3 | Cumulative Credit, Incremental Equivalent Debt, Excess Cash Flow, Effective Yield | 0 |
| [digitalocean-1582961](https://www.sec.gov/Archives/edgar/data/1582961/000158296125000084/creditagreement.htm) | DigitalOcean Holdings, Inc.  (DOCN)  (CI | 2025-05-05 | 714,992 | leveraged | test | 3 | Incremental Equivalent Debt, Ratio Debt | 1 |
| [esco-866706](https://www.sec.gov/Archives/edgar/data/866706/000110465926070116/tm2616691d1_ex10-1.htm) | ESCO TECHNOLOGIES INC  (ESE)  (CIK 00008 | 2026-05-29 | 795,331 | leveraged | train | 1 | Available Amount, Incremental Equivalent Debt, Excess Cash Flow, Effective Yield | 0 |
| [eventbrite-1475115](https://www.sec.gov/Archives/edgar/data/1475115/000147511525000105/creditagreementdatedaugust.htm) | Eventbrite, Inc.  (EB)  (CIK 0001475115) | 2025-08-06 | 702,075 | leveraged | train | 3 | Excess Cash Flow, Effective Yield, All-In Yield | 0 |
| [everus-2015845](https://www.sec.gov/Archives/edgar/data/2015845/000114036124044886/ef20037700_ex10-4.htm) | Everus Construction Group, Inc.  (ECG, E | 2024-10-31 | 964,944 | leveraged | test | 3 | Incremental Equivalent Debt, Term B Loans, Ratio Debt | 0 |
| [evgo-1821159](https://www.sec.gov/Archives/edgar/data/1821159/000110465925109155/evgo-20250930xex10d1.htm) | EVgo Inc.  (EVGO, EVGOW)  (CIK 000182115 | 2025-07-23 | 463,293 | leveraged | train | 3 | Available Amount, Excess Cash Flow | 0 |
| [evolent-1628908](https://www.sec.gov/Archives/edgar/data/1628908/000162890825000116/a6302025exhibit103.htm) | Evolent Health, Inc.  (EVH)  (CIK 000162 | 2025-08-07 | 496,967 | leveraged | train | 4 | Available Amount, Excess Cash Flow | 0 |
| [gogo-1537054](https://www.sec.gov/Archives/edgar/data/1537054/000119312524273399/d915976dex102.htm) | Gogo Inc.  (GOGO)  (CIK 0001537054) | 2024-12-03 | 709,175 | leveraged | train | 3 | Available Amount, Incremental Equivalent Debt, Excess Cash Flow, Effective Yield | 0 |
| [health-1636422](https://www.sec.gov/Archives/edgar/data/1636422/000163642224000120/exhibit101healthcatalyst-c.htm) | Health Catalyst, Inc.  (HCAT)  (CIK 0001 | 2024-07-16 | 569,745 | leveraged | train | 2 | Available Amount, Excess Cash Flow, All-In Yield | 0 |
| [herc-1364479](https://www.sec.gov/Archives/edgar/data/1364479/000119312525133360/d943577dex104.htm) | HERC HOLDINGS INC  (HRI)  (CIK 000136447 | 2025-06-02 | 605,777 | leveraged | train | 1 | Cumulative Credit, Incremental Equivalent Debt | 1 |
| [hillman-1822492](https://www.sec.gov/Archives/edgar/data/1822492/000182249226000122/ex102-hillmanxablcreditagr.htm) | Hillman Solutions Corp.  (HLMN)  (CIK 00 | 2026-07-22 | 873,100 | leveraged | train | 1 | Available Amount, Incremental Equivalent Debt | 0 |
| [huntsman-1307954](https://www.sec.gov/Archives/edgar/data/1307954/000110465926015235/tm265738d1_ex10-1.htm) | Huntsman CORP  (HUN)  (CIK 0001307954) | 2026-02-09 | 576,164 | leveraged | test | 3 | Available Amount, Incremental Equivalent Debt | 2 |
| [iheartmedia-1400891](https://www.sec.gov/Archives/edgar/data/1400891/000119312524283675/d854532dex102.htm) | iHeartMedia, Inc.  (IHRT, IHETW, IHRTB)  | 2024-12-20 | 724,340 | leveraged | train | 3 | Excess Cash Flow, All-In Yield | 0 |
| [kore-1855457](https://www.sec.gov/Archives/edgar/data/1855457/000119312523274584/d318950dex102.htm) | KORE Group Holdings, Inc.  (KORE, KORGW) | 2023-11-09 | 814,042 | leveraged | train | 3 | Available Amount, Incremental Equivalent Debt, Excess Cash Flow | 1 |
| [krispy-1857154](https://www.sec.gov/Archives/edgar/data/1857154/000185715425000131/a109-creditagreementdatedm.htm) | Krispy Kreme, Inc.  (DNUT)  (CIK 0001857 | 2023-03-23 | 657,822 | leveraged | train | 3 | Available Amount, Incremental Equivalent Debt, Excess Cash Flow | 1 |
| [lands-799288](https://www.sec.gov/Archives/edgar/data/799288/000095017024001005/le-ex4_1.htm) | LANDS' END, INC.  (LE)  (CIK 0000799288) | 2023-12-29 | 484,045 | leveraged | train | 1 | Excess Cash Flow, Effective Yield | 0 |
| [lci-763744](https://www.sec.gov/Archives/edgar/data/763744/000076374425000068/lcii-03312025xex101.htm) | LCI INDUSTRIES  (LCII)  (CIK 0000763744) | 2025-03-25 | 729,578 | leveraged | test | 3 | Available Amount, Incremental Equivalent Debt, Excess Cash Flow, Effective Yield | 0 |
| [light-750004](https://www.sec.gov/Archives/edgar/data/750004/000110465925050011/tm2515308d1_ex10-1.htm) | Light & Wonder, Inc.  (LNW, LAWIL)  (CIK | 2025-05-15 | 573,448 | leveraged | train | 3 | Available Amount, Excess Cash Flow | 0 |
| [lumen-18926](https://www.sec.gov/Archives/edgar/data/18926/000119312526157915/d144157dex101.htm) | Lumen Technologies, Inc.  (LUMN)  (CIK 0 | 2026-04-14 | 672,530 | leveraged | test | 3 | Incremental Equivalent Debt, Excess Cash Flow | 0 |
| [lumentum-1633978](https://www.sec.gov/Archives/edgar/data/1633978/000119312525328691/d38670dex101.htm) | Lumentum Holdings Inc.  (LITE)  (CIK 000 | 2025-12-19 | 679,240 | leveraged | train | 3 | Incremental Equivalent Debt, Ratio Debt | 1 |
| [maravai-1823239](https://www.sec.gov/Archives/edgar/data/1823239/000182323926000049/newcreditagreement.htm) | MARAVAI LIFESCIENCES HOLDINGS, INC.  (MR | 2026-06-02 | 925,968 | leveraged | train | 3 | Available Amount, Builder Basket, Incremental Equivalent Debt, Excess Cash Flow, Effective Yield | 0 |
| [masterbrand-1941365](https://www.sec.gov/Archives/edgar/data/1941365/000114036124031587/ef20031782_ex10-1.htm) | MasterBrand, Inc.  (MBC)  (CIK 000194136 | 2024-06-27 | 514,813 | leveraged | train | 2 | Term B Loans, Ratio Debt | 0 |
| [mp-1801368](https://www.sec.gov/Archives/edgar/data/1801368/000119312525187776/d15553dex101.htm) | MP Materials Corp. / DE  (MP)  (CIK 0001 | 2025-08-25 | 697,968 | leveraged | train | 3 | Incremental Equivalent Debt, Ratio Debt | 1 |
| [national-1377630](https://www.sec.gov/Archives/edgar/data/1377630/000119312526396098/ncmi-ex10_1.htm) | National CineMedia, Inc.  (NCMI)  (CIK 0 | 2026-09-18 | 753,697 | leveraged | train | 3 | Cumulative Credit, Excess Cash Flow, Effective Yield | 0 |
| [neurocrine-914475](https://www.sec.gov/Archives/edgar/data/914475/000119312526228013/d94926dex101.htm) | NEUROCRINE BIOSCIENCES INC  (NBIX)  (CIK | 2026-05-14 | 702,086 | leveraged | train | 2 | Available Amount, Incremental Equivalent Debt | 0 |
| [on-1097864](https://www.sec.gov/Archives/edgar/data/1097864/000119312523174498/d476071dex101.htm) | ON SEMICONDUCTOR CORP  (ON)  (CIK 000109 | 2023-06-22 | 575,420 | leveraged | train | 3 | Available Amount, Incremental Equivalent Indebtedness, Effective Yield | 2 |
| [one-1399520](https://www.sec.gov/Archives/edgar/data/1399520/000155837024006331/stks-20240501xex10d1.htm) | ONE Group Hospitality, Inc.  (STKS)  (CI | 2024-05-01 | 730,577 | leveraged | test | 2 | Available Amount, Excess Cash Flow, Effective Yield | 0 |
| [outfront-1579877](https://www.sec.gov/Archives/edgar/data/1579877/000157987725000113/creditagt.htm) | OUTFRONT Media Inc.  (OUT)  (CIK 0001579 | 2025-09-24 | 804,615 | leveraged | test | 3 | Available Amount, All-In Yield | 1 |
| [pitney-78814](https://www.sec.gov/Archives/edgar/data/78814/000007881425000007/pitneybowes-creditagreemen.htm) | PITNEY BOWES INC /DE/  (PBI, PBI-PB)  (C | 2025-02-07 | 747,237 | leveraged | train | 2 | Available Amount, Excess Cash Flow | 0 |
| [ranpak-1712463](https://www.sec.gov/Archives/edgar/data/1712463/000095010324018007/dp222355_ex1001.htm) | Ranpak Holdings Corp.  (PACK)  (CIK 0001 | 2024-12-19 | 993,905 | leveraged | train | 1 | Available Amount, Incremental Equivalent Debt, Excess Cash Flow, Effective Yield | 0 |
| [rayonier-1597672](https://www.sec.gov/Archives/edgar/data/1597672/000114036124044534/ef20037818_ex10-1.htm) | RAYONIER ADVANCED MATERIALS INC.  (RYAM) | 2024-10-28 | 566,817 | leveraged | train | 2 | Cumulative Credit, Excess Cash Flow | 0 |
| [red-1653653](https://www.sec.gov/Archives/edgar/data/1653653/000119312524067880/d806794dex101.htm) | Red Rock Resorts, Inc.  (RRR)  (CIK 0001 | 2024-03-14 | 985,685 | leveraged | train | 3 | Available Amount, Excess Cash Flow, All-In Yield, Ratio Debt | 0 |
| [remitly-1782170](https://www.sec.gov/Archives/edgar/data/1782170/000095010325008001/dp230566_ex1001.htm) | Remitly Global, Inc.  (RELY)  (CIK 00017 | 2025-06-24 | 597,971 | leveraged | train | 3 | Available Amount, Incremental Equivalent Debt, Ratio Debt | 0 |
| [ribbon-1708055](https://www.sec.gov/Archives/edgar/data/1708055/000110465924074398/tm2417775d1_ex10-1.htm) | Ribbon Communications Inc.  (RBBN)  (CIK | 2024-06-21 | 794,422 | leveraged | test | 2 | Available Amount, Excess Cash Flow, All-In Yield | 0 |
| [rite-84129](https://www.sec.gov/Archives/edgar/data/84129/000110465923129388/tm2333722d1_ex10-1.htm) | RITE AID CORP  (RADCQ)  (CIK 0000084129) | 2023-11-08 | 608,062 | leveraged | train | 2 | Excess Cash Flow, Effective Yield | 0 |
| [slide-1886428](https://www.sec.gov/Archives/edgar/data/1886428/000119312526434282/slde-ex10_1.htm) | Slide Insurance Holdings, Inc.  (SLDE)   | 2026-10-26 | 711,774 | leveraged | train | 3 | Excess Cash Flow, All-In Yield | 1 |
| [solv-2065636](https://www.sec.gov/Archives/edgar/data/2065636/000119312526048814/d189834dex106.htm) | SOLV Energy, Inc.  (MWH)  (CIK 000206563 | 2026-02-12 | 797,346 | leveraged | test | 2 | Available Amount, Incremental Equivalent Debt, Ratio Debt | 0 |
| [venture-2007855](https://www.sec.gov/Archives/edgar/data/2007855/000200785526000062/exhibit101-q22026.htm) | Venture Global, Inc.  (VG)  (CIK 0002007 | 2026-04-10 | 553,094 | leveraged | train | 3 | Incremental Equivalent Debt, Excess Cash Flow, All-In Yield | 0 |
| [walker-1497770](https://www.sec.gov/Archives/edgar/data/1497770/000110465925023966/tm258986d1_ex10-1.htm) | Walker & Dunlop, Inc.  (WD)  (CIK 000149 | 2025-03-14 | 659,145 | leveraged | train | 2 | Available Amount, Incremental Equivalent Debt, Excess Cash Flow, Effective Yield | 1 |
| [wayfair-1616707](https://www.sec.gov/Archives/edgar/data/1616707/000119312525054415/d909268dex101.htm) | Wayfair Inc.  (W)  (CIK 0001616707) | 2025-03-13 | 787,561 | leveraged | train | 3 | Available Amount, Effective Yield | 1 |
| [worthington-1968487](https://www.sec.gov/Archives/edgar/data/1968487/000196848726000018/ws-ex10_1.htm) | Worthington Steel, Inc.  (WS)  (CIK 0001 | 2026-06-25 | 746,649 | leveraged | train | 1 | Incremental Equivalent Debt, Ratio Debt | 0 |
| [xpo-1166003](https://www.sec.gov/Archives/edgar/data/1166003/000110465926068522/tm2616097d1_ex10-2.htm) | XPO, Inc.  (XPO)  (CIK 0001166003) | 2026-05-29 | 565,718 | leveraged | train | 2 | Cumulative Credit, Ratio Debt | 1 |
| [advance-1158449](https://www.sec.gov/Archives/edgar/data/1158449/000115844925000268/aapex101-ablcreditagreement.htm) | ADVANCE AUTO PARTS INC  (AAP)  (CIK 0001 | 2025-08-12 | 636,521 | secured_mid_cap | train | 2 |  | 0 |
| [aevex-2096300](https://www.sec.gov/Archives/edgar/data/2096300/000119312526177298/d108131dex101.htm) | AEVEX Corp.  (AVEX)  (CIK 0002096300) | 2026-04-20 | 992,788 | secured_mid_cap | train | 3 |  | 1 |
| [american-5272](https://www.sec.gov/Archives/edgar/data/5272/000000527224000126/q32024exhibit102.htm) | AMERICAN INTERNATIONAL GROUP, INC.  (AIG | 2024-09-27 | 441,287 | secured_mid_cap | train | 1 |  | 1 |
| [amphastar-1297184](https://www.sec.gov/Archives/edgar/data/1297184/000129718423000066/amph-20230630xex10d4.htm) | Amphastar Pharmaceuticals, Inc.  (AMPH)  | 2023-06-30 | 687,336 | secured_mid_cap | test | 2 | Incremental Equivalent Debt | 0 |
| [andersen-2065708](https://www.sec.gov/Archives/edgar/data/2065708/000119312526288725/d31545dex101.htm) | Andersen Group Inc.  (ANDG)  (CIK 000206 | 2026-06-25 | 513,748 | secured_mid_cap | train | 2 |  | 0 |
| [aris-1865187](https://www.sec.gov/Archives/edgar/data/1865187/000114036123048054/ef20012425_ex10-1.htm) | Aris Water Solutions, Inc.  (ARIS)  (CIK | 2023-10-12 | 568,692 | secured_mid_cap | train | 3 |  | 0 |
| [ast-1780312](https://www.sec.gov/Archives/edgar/data/1780312/000149315225011288/ex10-1.htm) | AST SpaceMobile, Inc.  (ASTS)  (CIK 0001 | 2025-07-15 | 354,497 | secured_mid_cap | train | 3 |  | 0 |
| [atricure-1323885](https://www.sec.gov/Archives/edgar/data/1323885/000119312524003922/d692835dex101.htm) | AtriCure, Inc.  (ATRC)  (CIK 0001323885) | 2024-01-05 | 578,452 | secured_mid_cap | train | 2 |  | 0 |
| [aviat-1377789](https://www.sec.gov/Archives/edgar/data/1377789/000119312523138929/d312982dex101.htm) | AVIAT NETWORKS, INC.  (AVNW)  (CIK 00013 | 2023-05-09 | 626,341 | secured_mid_cap | train | 2 |  | 0 |
| [avient-1122976](https://www.sec.gov/Archives/edgar/data/1122976/000119312525141088/d79513dex101.htm) | AVIENT CORP  (AVNT)  (CIK 0001122976) | 2025-06-12 | 594,602 | secured_mid_cap | train | 2 |  | 1 |
| [battalion-1282648](https://www.sec.gov/Archives/edgar/data/1282648/000110465926079612/batl-20260630xex10d1.htm) | BATTALION OIL CORP  (BATL)  (CIK 0001282 | 2026-06-30 | 585,057 | secured_mid_cap | train | 3 | Excess Cash Flow | 0 |
| [beyond-1130713](https://www.sec.gov/Archives/edgar/data/1130713/000114036125018409/ef20048777_ex10-1.htm) | BEYOND, INC.  (BYON)  (CIK 0001130713) | 2025-05-07 | 410,671 | secured_mid_cap | test | 3 |  | 0 |
| [bowman-1847590](https://www.sec.gov/Archives/edgar/data/1847590/000162828024021187/bwmn-20240331xexx1018credi.htm) | Bowman Consulting Group Ltd.  (BWMN)  (C | 2024-05-02 | 489,398 | secured_mid_cap | train | 2 |  | 0 |
| [buzzfeed-1828972](https://www.sec.gov/Archives/edgar/data/1828972/000182897225000139/exhibit101termloanexecutio.htm) | BuzzFeed, Inc.  (BZFD, BZFDW)  (CIK 0001 | 2025-05-23 | 358,303 | secured_mid_cap | train | 2 |  | 0 |
| [camping-1669779](https://www.sec.gov/Archives/edgar/data/1669779/000110465926103869/cwh-20260825xex10d1.htm) | Camping World Holdings, Inc.  (CWH)  (CI | 2026-08-25 | 540,711 | secured_mid_cap | train | 2 |  | 0 |
| [carecloud-1582982](https://www.sec.gov/Archives/edgar/data/1582982/000149315226016464/ex10-1.htm) | CareCloud, Inc.  (CCLD, CCLDO)  (CIK 000 | 2026-04-13 | 529,831 | secured_mid_cap | train | 2 | Effective Yield | 0 |
| [cavco-278166](https://www.sec.gov/Archives/edgar/data/278166/000027816624000161/cvco-20241113exhibit101.htm) | CAVCO INDUSTRIES INC.  (CVCO)  (CIK 0000 | 2024-11-12 | 475,708 | secured_mid_cap | train | 2 |  | 0 |
| [cbiz-944148](https://www.sec.gov/Archives/edgar/data/944148/000119312524249818/d883302dex101.htm) | CBIZ, Inc.  (CBZ)  (CIK 0000944148) | 2024-11-01 | 500,995 | secured_mid_cap | train | 2 |  | 0 |
| [cnx-1070412](https://www.sec.gov/Archives/edgar/data/1070412/000119312524141817/d773002dex102.htm) | CNX Resources Corp  (CNX)  (CIK 00010704 | 2024-05-17 | 575,369 | secured_mid_cap | train | 3 |  | 1 |
| [coeur-215466](https://www.sec.gov/Archives/edgar/data/215466/000114036126010650/ef20066748_ex10-1.htm) | Coeur Mining, Inc.  (CDE)  (CIK 00002154 | 2026-03-20 | 555,219 | secured_mid_cap | test | 1 |  | 1 |
| [conduent-1677703](https://www.sec.gov/Archives/edgar/data/1677703/000167770326000142/exhibit103-conduentxinterc.htm) | CONDUENT Inc  (CNDT)  (CIK 0001677703) | 2026-10-23 | 626,474 | secured_mid_cap | test | 3 |  | 0 |
| [csw-1624794](https://www.sec.gov/Archives/edgar/data/1624794/000119312525264609/d47476dex101.htm) | CSW INDUSTRIALS, INC.  (CSW)  (CIK 00016 | 2025-11-04 | 628,259 | secured_mid_cap | train | 2 |  | 0 |
| [duluth-1649744](https://www.sec.gov/Archives/edgar/data/1649744/000119312525110141/d933264dex101.htm) | DULUTH HOLDINGS INC.  (DLTH)  (CIK 00016 | 2025-04-28 | 636,116 | secured_mid_cap | train | 1 |  | 0 |
| [energy-1828536](https://www.sec.gov/Archives/edgar/data/1828536/000182853626000106/evgensetcreditagreement-ex.htm) | Energy Vault Holdings, Inc.  (NRGV)  (CI | 2026-08-14 | 404,889 | secured_mid_cap | train | 1 |  | 0 |
| [enerpac-6955](https://www.sec.gov/Archives/edgar/data/6955/000114036126040100/ef20084020_ex10-1.htm) | ENERPAC TOOL GROUP CORP  (EPAC)  (CIK 00 | 2026-10-16 | 747,329 | secured_mid_cap | test | 3 |  | 0 |
| [enfusion-1868912](https://www.sec.gov/Archives/edgar/data/1868912/000155837023015842/enfn-20230915xex10d1.htm) | Enfusion, Inc.  (ENFN)  (CIK 0001868912) | 2023-09-15 | 512,887 | secured_mid_cap | train | 2 | Incremental Equivalent Debt | 0 |
| [etsy-1370637](https://www.sec.gov/Archives/edgar/data/1370637/000137063723000049/ex105q12023.htm) | ETSY INC  (ETSY)  (CIK 0001370637) | 2023-03-24 | 448,018 | secured_mid_cap | train | 2 |  | 0 |
| [greif-43920](https://www.sec.gov/Archives/edgar/data/43920/000162828026015049/gef2026creditagreementex10.htm) | GREIF, INC  (GEF, GEF-B)  (CIK 000004392 | 2026-02-27 | 738,585 | secured_mid_cap | train | 2 | Incremental Equivalent Debt | 1 |
| [icf-1362004](https://www.sec.gov/Archives/edgar/data/1362004/000143774926012565/ex_945516.htm) | ICF International, Inc.  (ICFI)  (CIK 00 | 2026-04-10 | 604,404 | secured_mid_cap | train | 3 | Incremental Equivalent Debt | 0 |
| [kestra-1877184](https://www.sec.gov/Archives/edgar/data/1877184/000119312526303397/ck0001877184-ex10_4.htm) | KESTRA MEDICAL TECHNOLOGIES, LTD.  (KMTS | 2026-07-10 | 558,328 | secured_mid_cap | train | 3 |  | 0 |
| [lsi-763532](https://www.sec.gov/Archives/edgar/data/763532/000143774926009635/ex_936252.htm) | LSI INDUSTRIES INC  (LYTS)  (CIK 0000763 | 2026-03-24 | 456,163 | secured_mid_cap | train | 2 |  | 0 |
| [macerich-912242](https://www.sec.gov/Archives/edgar/data/912242/000119312526077466/d112967dex101.htm) | MACERICH CO  (MAC)  (CIK 0000912242) | 2026-02-24 | 666,897 | secured_mid_cap | test | 2 |  | 0 |
| [merit-856982](https://www.sec.gov/Archives/edgar/data/856982/000085698223000049/mmsi-20230630xex10d2.htm) | MERIT MEDICAL SYSTEMS INC  (MMSI)  (CIK  | 2023-06-06 | 618,158 | secured_mid_cap | test | 2 |  | 0 |
| [mgp-835011](https://www.sec.gov/Archives/edgar/data/835011/000162828025020711/ex101arcafor8-k.htm) | MGP INGREDIENTS INC  (MGPI)  (CIK 000083 | 2025-04-24 | 512,000 | secured_mid_cap | test | 3 | Incremental Equivalent Debt | 0 |
| [midera-2088281](https://www.sec.gov/Archives/edgar/data/2088281/000119312526288554/d70874dex101.htm) | Midera Food Processing, Inc.  (MFP)  (CI | 2026-06-29 | 830,667 | secured_mid_cap | train | 3 | Incremental Equivalent Debt | 0 |
| [moneylion-1807846](https://www.sec.gov/Archives/edgar/data/1807846/000121390024102284/ea022247101ex10-1_moneylion.htm) | MONEYLION INC.  (ML, ML-WT)  (CIK 000180 | 2024-11-25 | 542,399 | secured_mid_cap | train | 2 | Incremental Equivalent Debt | 0 |
| [nuvera-71557](https://www.sec.gov/Archives/edgar/data/71557/000151316224000083/exhibit10_1.htm) | Nuvera Communications, Inc.  (NUVR)  (CI | 2024-06-21 | 535,036 | secured_mid_cap | train | 2 |  | 0 |
| [one-712770](https://www.sec.gov/Archives/edgar/data/712770/000110465926091179/olp-20260630xex10d3.htm) | ONE LIBERTY PROPERTIES INC  (OLP)  (CIK  | 2026-07-31 | 443,395 | secured_mid_cap | train | 1 |  | 1 |
| [orion-1402829](https://www.sec.gov/Archives/edgar/data/1402829/000140282925000065/orn-20251223xex10d1.htm) | Orion Group Holdings Inc  (ORN)  (CIK 00 | 2025-12-23 | 499,729 | secured_mid_cap | train | 2 |  | 0 |
| [pbf-1534504](https://www.sec.gov/Archives/edgar/data/1534504/000153450426000033/a101amendedandrestatedseni.htm) | PBF Energy Inc.  (PBF)  (CIK 0001534504) | 2026-09-30 | 622,981 | secured_mid_cap | train | 3 |  | 0 |
| [pershing-2026053](https://www.sec.gov/Archives/edgar/data/2026053/000114036126018597/ny20040230x31_ex10-4.htm) | PERSHING SQUARE INC.  (PS)  (CIK 0002026 | 2026-04-30 | 503,762 | secured_mid_cap | train | 2 | Incremental Equivalent Indebtedness | 0 |
| [phreesia-1412408](https://www.sec.gov/Archives/edgar/data/1412408/000141240826000057/ex101-capitaloneagreement.htm) | Phreesia, Inc.  (PHR)  (CIK 0001412408) | 2026-03-13 | 561,137 | secured_mid_cap | test | 1 | All-In Yield | 0 |
| [purple-1643953](https://www.sec.gov/Archives/edgar/data/1643953/000121390024005735/ea192025ex10-1_purple.htm) | Purple Innovation, Inc.  (PRPL)  (CIK 00 | 2024-01-23 | 481,034 | secured_mid_cap | train | 1 |  | 0 |
| [rmr-1644378](https://www.sec.gov/Archives/edgar/data/1644378/000110465925005461/tm254142d1_ex10-1.htm) | RMR GROUP INC.  (RMR)  (CIK 0001644378) | 2025-01-22 | 439,782 | secured_mid_cap | train | 3 |  | 2 |
| [root-1788882](https://www.sec.gov/Archives/edgar/data/1788882/000178888226000046/a101creditagreement.htm) | Root, Inc.  (ROOT)  (CIK 0001788882) | 2026-05-04 | 515,782 | secured_mid_cap | train | 1 |  | 0 |
| [scotts-825542](https://www.sec.gov/Archives/edgar/data/825542/000082554225000018/exhibit101smg2025-11x248xk.htm) | SCOTTS MIRACLE-GRO CO  (SMG)  (CIK 00008 | 2025-11-21 | 797,661 | secured_mid_cap | train | 1 |  | 0 |
| [sphere-1795250](https://www.sec.gov/Archives/edgar/data/1795250/000119312526035571/d76310dex101.htm) | Sphere Entertainment Co.  (SPHR)  (CIK 0 | 2026-01-29 | 521,962 | secured_mid_cap | train | 2 | Incremental Equivalent Debt | 0 |
| [sprout-1517375](https://www.sec.gov/Archives/edgar/data/1517375/000151737523000092/sprout-creditagreement8xkt.htm) | Sprout Social, Inc.  (SPT)  (CIK 0001517 | 2023-08-01 | 547,207 | secured_mid_cap | train | 2 |  | 0 |
| [stagwell-876883](https://www.sec.gov/Archives/edgar/data/876883/000087688326000010/stgw12312025-exhibit102.htm) | Stagwell Inc  (STGW)  (CIK 0000876883) | 2025-04-23 | 596,664 | secured_mid_cap | train | 2 |  | 0 |
| [star-1002590](https://www.sec.gov/Archives/edgar/data/1002590/000095017024110563/sgu-ex10_10.htm) | STAR GROUP, L.P.  (SGU)  (CIK 0001002590 | 2024-09-27 | 507,626 | secured_mid_cap | train | 3 | Excess Cash Flow | 0 |
| [superior-95574](https://www.sec.gov/Archives/edgar/data/95574/000143774926026859/ex_1002549.htm) | SUPERIOR GROUP OF COMPANIES, INC.  (SGC) | 2026-08-07 | 491,933 | secured_mid_cap | train | 2 |  | 0 |
| [targa-1389170](https://www.sec.gov/Archives/edgar/data/1389170/000095017025023983/trgp-ex10_1.htm) | Targa Resources Corp.  (TRGP)  (CIK 0001 | 2025-02-18 | 453,915 | secured_mid_cap | test | 1 |  | 1 |
| [terawulf-1083301](https://www.sec.gov/Archives/edgar/data/1083301/000108330126000092/terawulf-revolvingcreditag.htm) | TERAWULF INC.  (WULF)  (CIK 0001083301) | 2026-05-07 | 584,144 | secured_mid_cap | train | 3 |  | 0 |
| [trisalus-1826667](https://www.sec.gov/Archives/edgar/data/1826667/000182666724000017/a101creditagreementdatedap.htm) | TriSalus Life Sciences, Inc.  (TLSI, TLS | 2024-04-30 | 322,698 | secured_mid_cap | train | 1 |  | 0 |
| [united-1067701](https://www.sec.gov/Archives/edgar/data/1067701/000110465925067406/tm2520569d1_ex10-1.htm) | UNITED RENTALS, INC.  (URI)  (CIK 000106 | 2025-07-10 | 862,145 | secured_mid_cap | test | 1 | Ratio Debt | 0 |
| [usana-896264](https://www.sec.gov/Archives/edgar/data/896264/000089626425000175/ex101thirdamendedandrestat.htm) | USANA HEALTH SCIENCES INC  (USNA)  (CIK  | 2025-06-27 | 515,585 | secured_mid_cap | train | 2 |  | 0 |
| [veris-924901](https://www.sec.gov/Archives/edgar/data/924901/000092490124000044/vre10-qexhibit1022.htm) | Veris Residential, Inc.  (VRE)  (CIK 000 | 2024-04-22 | 562,313 | secured_mid_cap | test | 1 |  | 0 |
| [vitesse-1944558](https://www.sec.gov/Archives/edgar/data/1944558/000119312523008895/d426246dex102.htm) | Vitesse Energy, Inc.  (VTS)  (CIK 000194 | 2023-01-13 | 483,145 | secured_mid_cap | train | 1 |  | 0 |
| [vivid-1856031](https://www.sec.gov/Archives/edgar/data/1856031/000119312526331507/seat-ex10_2.htm) | Vivid Seats Inc.  (SEAT, SEATW)  (CIK 00 | 2026-08-03 | 743,724 | secured_mid_cap | train | 3 | Incremental Equivalent Debt | 0 |
| [world-108385](https://www.sec.gov/Archives/edgar/data/108385/000010838525000089/worldacceptance-warehousec.htm) | WORLD ACCEPTANCE CORP  (WRLD)  (CIK 0000 | 2025-09-29 | 624,499 | secured_mid_cap | train | 1 | Available Amount | 0 |
