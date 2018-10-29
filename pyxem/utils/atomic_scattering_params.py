# Electron scattering factors from International Tables Vol. C, table 4.3.2.3
# name  a1       b1      a2      b2      a3      b3       a4       b4        a5      b5

ATOMIC_SCATTERING_PARAMS = dict((
    ('H', ((0.0088, 0.1152), (0.0449, 1.0867), (0.1481, 4.9755), (0.2356, 16.5591), (0.0914, 43.2743))),
    ('He', ((0.0084, 0.0596), (0.0443, 0.5360), (0.1314, 2.4274), (0.1671, 7.7852), (0.0666, 20.3126))),
    ('Li', ((0.0478, 0.2258), (0.2048, 2.1032), (0.5253, 12.9349), (1.5225, 50.7501), (0.9853, 136.6280))),
    ('Be', ((0.0423, 0.1445), (0.1874, 1.4180), (0.6019, 8.1165), (1.4311, 27.9705), (0.7891, 74.8684))),
    ('B', ((0.0436, 0.1207), (0.1898, 1.1595), (0.6788, 6.2474), (1.3273, 21.0460), (0.5544, 59.3619))),
    ('C', ((0.0489, 0.1140), (0.2091, 1.0825), (0.7537, 5.4281), (1.1420, 17.8811), (0.3555, 51.1341))),
    ('N', ((0.0267, 0.0541), (0.1328, 0.5165), (0.5301, 2.8207), (1.1020, 10.6297), (0.4215, 34.3764))),
    ('O', ((0.0365, 0.0652), (0.1729, 0.6184), (0.5805, 2.9449), (0.8814, 9.6298), (0.3121, 28.2194))),
    ('F', ((0.0382, 0.0613), (0.1822, 0.5753), (0.5972, 2.6858), (0.7707, 8.8214), (0.2130, 25.6668))),
    ('Ne', ((0.0380, 0.0554), (0.1785, 0.5087), (0.5494, 2.2639), (0.6942, 7.3316), (0.1918, 21.6912))),
    ('Na', ((0.1260, 0.1684), (0.6442, 1.7150), (0.8893, 8.8386), (1.8197, 50.8265), (1.2988, 147.2073))),
    ('Mg', ((0.1130, 0.1356), (0.5575, 1.3579), (0.9046, 6.9255), (2.1580, 32.3165), (1.4735, 92.1138))),
    ('Al', ((0.1165, 0.1295), (0.5504, 1.2619), (1.0179, 6.8242), (2.6295, 28.4577), (1.5711, 88.4750))),
    ('Si', ((0.0567, 0.0582), (0.3365, 0.6155), (0.8104, 3.2522), (2.4960, 16.7929), (2.1186, 57.6767))),
    ('P', ((0.1005, 0.0977), (0.4615, 0.9084), (1.0663, 4.9654), (2.5854, 18.5471), (1.2725, 54.3648))),
    ('S', ((0.0915, 0.0838), (0.4312, 0.7788), (1.0847, 4.3462), (2.4671, 15.5846), (1.0852, 44.6365))),
    ('Cl', ((0.0799, 0.0694), (0.3891, 0.6443), (1.0037, 3.5351), (2.3332, 12.5058), (1.0507, 35.8633))),
    ('Ar', ((0.1044, 0.0853), (0.4551, 0.7701), (1.4232, 4.4684), (2.1533, 14.5864), (0.4459, 41.2474))),
    ('K', ((0.2149, 0.1660), (0.8703, 1.6906), (2.4999, 8.7447), (2.3591, 46.7825), (3.0318, 165.6923))),
    ('Ca', ((0.2355, 0.1742), (0.9916, 1.8329), (2.3959, 8.8407), (3.7252, 47.4583), (2.5647, 134.9613))),
    ('Sc', ((0.4636, 0.3682), (2.0802, 4.0312), (2.9003, 22.6493), (1.4193, 71.8200), (2.4323, 103.3691))),
    ('Ti', ((0.2123, 0.1399), (0.8960, 1.4568), (2.1765, 6.7534), (3.0436, 33.1168), (2.4439, 101.8238))),
    ('V', ((0.2369, 0.1505), (1.0774, 1.6392), (2.1894, 7.5691), (3.0825, 36.8741), (1.7190, 107.8517))),
    ('Cr', ((0.1970, 0.1197), (0.8228, 1.1985), (2.0200, 5.4097), (2.1717, 25.2361), (1.7516, 94.4290))),
    ('Mn', ((0.1943, 0.1135), (0.8190, 1.1313), (1.9296, 5.0341), (2.4968, 24.1798), (2.0625, 80.5598))),
    ('Fe', ((0.1929, 0.1087), (0.8239, 1.0806), (1.8689, 4.7637), (2.3694, 22.8500), (1.9060, 76.7309))),
    ('Co', ((0.2186, 0.1182), (0.9861, 1.2300), (1.8540, 5.4177), (2.3258, 25.7602), (1.4685, 80.8542))),
    ('Ni', ((0.2313, 0.1210), (1.0657, 1.2691), (1.8229, 5.6870), (2.2609, 27.0917), (1.1883, 83.0285))),
    ('Cu', ((0.3501, 0.1867), (1.6558, 1.9917), (1.9582, 11.3396), (0.2134, 53.2619), (1.4109, 63.2520))),
    ('Zn', ((0.1780, 0.0876), (0.8096, 0.8650), (1.6744, 3.8612), (1.9499, 18.8726), (1.4495, 64.7016))),
    ('Ga', ((0.2135, 0.1020), (0.9768, 1.0219), (1.6669, 4.6275), (2.5662, 22.8742), (1.6790, 80.1535))),
    ('Ge', ((0.2135, 0.0989), (0.9761, 0.9845), (1.6555, 4.5527), (2.8938, 21.5563), (1.6356, 70.3903))),
    ('As', ((0.2059, 0.0926), (0.9518, 0.9182), (1.6372, 4.3291), (3.0490, 19.2996), (1.4756, 58.9329))),
    ('Se', ((0.1574, 0.0686), (0.7614, 0.6808), (1.4834, 3.1163), (3.0016, 14.3458), (1.7978, 44.0455))),
    ('Br', ((0.1899, 0.0810), (0.8983, 0.7957), (1.6358, 3.9054), (3.1845, 15.7701), (1.1518, 45.6124))),
    ('Kr', ((0.1742, 0.0723), (0.8447, 0.7123), (1.5944, 3.5192), (3.1507, 13.7724), (1.1338, 39.1148))),
    ('Rb', ((0.3781, 0.1557), (1.4904, 1.5347), (3.5753, 9.9947), (3.0031, 51.4251), (3.3272, 185.9828))),
    ('Sr', ((0.3723, 0.1480), (1.4598, 1.4643), (3.5124, 9.2320), (4.4612, 49.8807), (3.3031, 148.0937))),
    ('Y', ((0.3234, 0.1244), (1.2737, 1.1948), (3.2115, 7.2756), (4.0563, 34.1430), (3.7962, 111.2079))),
    ('Zr', ((0.2997, 0.1121), (1.1879, 1.0638), (3.1075, 6.3891), (3.9740, 28.7081), (3.5769, 97.4289))),
    ('Nb', ((0.1680, 0.0597), (0.9370, 0.6524), (2.7300, 4.4317), (3.8150, 19.5540), (3.0053, 85.5011))),
    ('Mo', ((0.3069, 0.1101), (1.1714, 1.0222), (3.2293, 5.9613), (3.4254, 25.1965), (2.1224, 93.5831))),
    ('Tc', ((0.2928, 0.1020), (1.1267, 0.9481), (3.1675, 5.4713), (3.6619, 23.8153), (2.5942, 82.8991))),
    ('Ru', ((0.2604, 0.0887), (1.0442, 0.8240), (3.0761, 4.8278), (3.2175, 19.8977), (1.9448, 80.4566))),
    ('Rh', ((0.2713, 0.0907), (1.0556, 0.8324), (3.1416, 4.7702), (3.0451, 19.7862), (1.7179, 80.2540))),
    ('Pd', ((0.2003, 0.0659), (0.8779, 0.6111), (2.6135, 3.5563), (2.8594, 12.7638), (1.0258, 44.4283))),
    ('Ag', ((0.2739, 0.0881), (1.0503, 0.8028), (3.1564, 4.4451), (2.7543, 18.7011), (1.4328, 79.2633))),
    ('Cd', ((0.3072, 0.0966), (1.1303, 0.8856), (3.2046, 4.6273), (2.9329, 20.6789), (1.6560, 73.4723))),
    ('In', ((0.3564, 0.1091), (1.3011, 1.0452), (3.2424, 5.0900), (3.4839, 24.6578), (2.0459, 88.0513))),
    ('Sn', ((0.2966, 0.0896), (1.1157, 0.8268), (3.0973, 4.2242), (3.8156, 20.6900), (2.5281, 71.3399))),
    ('Sb', ((0.2725, 0.0809), (1.0651, 0.7488), (2.9940, 3.8710), (4.0697, 18.8800), (2.5682, 60.6499))),
    ('Te', ((0.2422, 0.0708), (0.9692, 0.6472), (2.8114, 3.3609), (4.1509, 16.0752), (2.8161, 50.1724))),
    ('I', ((0.2617, 0.0749), (1.0325, 0.6914), (2.8097, 3.4634), (4.4809, 16.3603), (2.3190, 48.2522))),
    ('Xe', ((0.2334, 0.0655), (0.9496, 0.6050), (2.6381, 3.0389), (4.4680, 14.0809), (2.5020, 41.0005))),
    ('Cs', ((0.5713, 0.1626), (2.4866, 1.8213), (4.9795, 11.1049), (4.0198, 49.0568), (4.4403, 202.9987))),
    ('Ba', ((0.5229, 0.1434), (2.2874, 1.6019), (4.7243, 9.4511), (5.0807, 42.7685), (5.6389, 148.4969))),
    ('La', ((0.5461, 0.1479), (2.3856, 1.6552), (5.0653, 10.0059), (5.7601, 47.3245), (4.0463, 145.8464))),
    ('Ce', ((0.2227, 0.0571), (1.0760, 0.5946), (2.9482, 3.2022), (5.8496, 16.4253), (7.1834, 95.7030))),
    ('Pr', ((0.5237, 0.1360), (2.2913, 1.5068), (4.6161, 8.8213), (4.7233, 41.9536), (4.8173, 141.2424))),
    ('Nd', ((0.5368, 0.1378), (2.3301, 1.5140), (4.6058, 8.8719), (4.6621, 43.5967), (4.4622, 141.8065))),
    ('Pm', ((0.5232, 0.1317), (2.2627, 1.4336), (4.4552, 8.3087), (4.4787, 40.6010), (4.5073, 135.9196))),
    ('Sm', ((0.5162, 0.1279), (2.2302, 1.3811), (4.3449, 7.9629), (4.3598, 39.1213), (4.4292, 132.7846))),
    ('Eu', ((0.5272, 0.1285), (2.2844, 1.3943), (4.3361, 8.1081), (4.3178, 40.9631), (4.0908, 134.1233))),
    ('Gd', ((0.9664, 0.2641), (3.4052, 2.6586), (5.0803, 16.2213), (1.4991, 80.2060), (4.2528, 92.5359))),
    ('Tb', ((0.5110, 0.1210), (2.1570, 1.2704), (4.0308, 7.1368), (3.9936, 35.0354), (4.2466, 123.5062))),
    ('Dy', ((0.4974, 0.1157), (2.1097, 1.2108), (3.8906, 6.7377), (3.8100, 32.4150), (4.3084, 116.9225))),
    ('Ho', ((0.4679, 0.1069), (1.9693, 1.0994), (3.7191, 5.9769), (3.9632, 27.1491), (4.2432, 96.3119))),
    ('Er', ((0.5034, 0.1141), (2.1088, 1.1769), (3.8232, 6.6087), (3.7299, 33.4332), (3.8963, 116.4913))),
    ('Tm', ((0.4839, 0.1081), (2.0262, 1.1012), (3.6851, 6.1114), (3.5874, 30.3728), (4.0037, 110.5988))),
    ('Yb', ((0.5221, 0.1148), (2.1695, 1.1860), (3.7567, 6.7520), (3.6685, 35.6807), (3.4274, 118.0692))),
    ('Lu', ((0.4680, 0.1015), (1.9466, 1.0195), (3.5428, 5.6058), (3.8490, 27.4899), (3.6594, 95.2846))),
    ('Hf', ((0.4048, 0.0868), (1.7370, 0.8585), (3.3399, 4.6378), (3.9448, 21.6900), (3.7293, 80.2408))),
    ('Ta', ((0.3835, 0.0810), (1.6747, 0.8020), (3.2986, 4.3545), (4.0462, 19.9644), (3.4303, 73.6337))),
    ('W', ((0.3661, 0.0761), (1.6191, 0.7543), (3.2455, 4.0952), (4.0856, 18.2886), (3.2064, 68.0967))),
    ('Re', ((0.3933, 0.0806), (1.6973, 0.7972), (3.4202, 4.4237), (4.1274, 19.5692), (2.6158, 68.7477))),
    ('Os', ((0.3854, 0.0787), (1.6555, 0.7638), (3.4129, 4.2441), (4.1111, 18.3700), (2.4106, 65.1071))),
    ('Ir', ((0.3510, 0.0706), (1.5620, 0.6904), (3.2946, 3.8266), (4.0615, 16.0812), (2.4382, 58.7638))),
    ('Pt', ((0.3083, 0.0609), (1.4158, 0.5993), (2.9662, 3.1921), (3.9349, 12.5285), (2.1709, 49.7675))),
    ('Au', ((0.3055, 0.0596), (1.3945, 0.5827), (2.9617, 3.1035), (3.8990, 11.9693), (2.0026, 47.9106))),
    ('Hg', ((0.3593, 0.0694), (1.5736, 0.6758), (3.5237, 3.8457), (3.8109, 15.6203), (1.6953, 56.6614))),
    ('Tl', ((0.3511, 0.0672), (1.5489, 0.6522), (3.5676, 3.7420), (4.0900, 15.9791), (2.5251, 65.1354))),
    ('Pb', ((0.3540, 0.0668), (1.5453, 0.6465), (3.5975, 3.6968), (4.3152, 16.2056), (2.7743, 61.4909))),
    ('Bi', ((0.3530, 0.0661), (1.5258, 0.6324), (3.5815, 3.5906), (4.5532, 15.9962), (3.0714, 57.5760))),
    ('Po', ((0.3673, 0.0678), (1.5772, 0.6527), (3.7079, 3.7396), (4.8582, 17.0668), (2.8440, 55.9789))),
    ('At', ((0.3547, 0.0649), (1.5206, 0.6188), (3.5621, 3.4696), (5.0184, 15.6090), (3.0075, 49.4818))),
    ('Rn', ((0.4586, 0.0831), (1.7781, 0.7840), (3.9877, 4.3599), (5.7273, 20.0128), (1.5460, 62.1535))),
    ('Fr', ((0.8282, 0.1515), (2.9941, 1.6163), (5.6597, 9.7752), (4.9292, 42.8480), (4.2889, 190.7366))),
    ('Ra', ((1.4129, 0.2921), (4.4269, 3.1381), (7.0460, 19.6767), (-1.0573, 102.0436), (8.6430, 113.9798))),
    ('Ac', ((0.7169, 0.1263), (2.5710, 1.2900), (5.1791, 7.3686), (6.3484, 32.4490), (5.6474, 118.0558))),
    ('Th', ((0.6958, 0.1211), (2.4936, 1.2247), (5.1269, 6.9398), (6.6988, 30.0991), (5.0799, 105.1960))),
    ('Pa', ((1.2502, 0.2415), (4.2284, 2.6442), (7.0489, 16.3313), (1.1390, 73.5757), (5.8222, 91.9401))),
    ('U', ((0.6410, 0.1097), (2.2643, 1.0644), (4.8713, 5.7907), (5.9287, 25.0261), (5.3935, 101.3899))),
    ('Np', ((0.6938, 0.1171), (2.4652, 1.1757), (5.1227, 6.4053), (5.5965, 27.5217), (4.8543, 103.0482))),
    ('Pu', ((0.6902, 0.1153), (2.4509, 1.1545), (5.1284, 6.2291), (5.0339, 27.0741), (4.8575, 111.3150))),
    ('Am', ((0.7577, 0.1257), (2.7264, 1.3044), (5.4184, 7.1035), (4.8198, 32.4649), (4.1013, 118.8647))),
    ('Cm', ((0.7567, 0.1239), (2.7565, 1.2979), (5.4364, 7.0798), (5.1918, 32.7871), (3.5643, 110.1512))),
    ('Bk', ((0.7492, 0.1217), (2.7267, 1.2651), (5.3521, 6.8101), (5.0369, 31.6088), (3.5321, 106.4853))),
    ('Cf', ((0.8100, 0.1310), (3.0001, 1.4038), (5.4635, 7.6057), (4.1756, 34.0186), (3.5066, 90.5226)))
))
