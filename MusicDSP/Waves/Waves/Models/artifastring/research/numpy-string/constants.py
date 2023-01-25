#!/usr/bin/env python


### this stuff ***must*** come before anything else!
import sys
which_string = sys.argv[-1]




import numpy
from defs import N

n = numpy.arange(1, N+1)

#which_string = "unit"
#which_string = "cello-c"
#which_string = "violin-e"
#which_string = "violin-g"

### simulation constants
if which_string == "violin-g":
    ### physical constants, G string
    T = 37.1        # tension, N
    L = 0.329       # length, m
    d = 0.8e-3      # diameter, m
    pl = 2.25e-3    # linear density, kg/m
    
    E = 4.0e9      # young's elastic modulus
    ##rn = numpy.array([0.48, 1.0, 1.8]) # measured empirically
    #rn = 0.5 + 0.5*(n)**2 # extremely vague approximation of modal decays
    rn = numpy.array([
6.754650202050435093e-01, 1.064581078186470409e+00,
1.578875957279010755e+00, 1.571492073584122462e+00,
2.388435484408834242e+00, 3.645865776718341333e+00,
4.889858529680899402e+00, 9.078039821405628373e+00,
1.328938832575954621e+01, 1.775752254823455445e+01,
2.454150018551774082e+01, 2.807004353068554181e+01,
3.091583011610751441e+01, 4.411384149310796943e+01,
4.822661826159256293e+01, 5.867402297935813493e+01,
7.061154881376222647e+01, 8.413853717258072606e+01,
9.935432946358953643e+01, 1.163582670945645674e+02,
1.352496914732817004e+02, 1.561279440075168736e+02,
1.790923661050459259e+02, 2.042422991736448239e+02,
2.316770846210894490e+02, 2.614960638551556826e+02,
2.937985782836194630e+02, 3.286839693142566148e+02,
3.662515783548431045e+02, 4.066007468131548421e+02,
4.498308160969676806e+02, 4.960411276140575296e+02,
5.453310227722002992e+02, 5.977998429791718991e+02,
6.535469296427482959e+02, 7.126716241707052859e+02,
7.752732679708187788e+02, 8.414512024508646846e+02,
9.113047690186190266e+02, 9.849333090818574874e+02,
1.062436164048356204e+03, 1.143912675325890859e+03,
1.229462184322237590e+03, 1.319184032445171852e+03,
1.413177561102470008e+03, 1.511542111701907970e+03,
1.614377025651261192e+03, 1.721781644358306039e+03,
1.833855309230818193e+03, 1.950697361676573564e+03,
2.072407143103348062e+03, 2.199083994918917597e+03,
2.330827258531058305e+03, 2.467736275347545416e+03,
2.609910386776155519e+03, 2.757448934224664299e+03,
2.910451259100847437e+03, 3.069016702812481071e+03,
3.233244606767341338e+03, 3.403234312373203466e+03,
3.579085161037844045e+03, 3.760896494169038760e+03,
3.948767653174562838e+03, 4.142797979462193325e+03,
                ])
elif which_string == "violin-e":
    ### physical constants, E string
    T = 81.8        # tension, N
    L = 0.329       # length, m
    d = 0.25e-3      # diameter, m
    pl = 4.34e-4    # linear density, kg/m
    
    E = 4.0e9      # young's elastic modulus
    #rn = 7.8 + 0.4*(n) # extremely vague approximation of modal decays
    rn = numpy.array([
2.729113295868674793e+00, 3.207715124938205875e+00,
8.386102768358499659e+00, 6.822461729010955267e+00,
1.052839892062496219e+01, 1.020277723825238070e+01,
1.470673700387809291e+01, 8.635839603366029849e+00,
9.479752968257109558e+00, 1.071331047909322010e+01,
9.913370750033188017e+00, 9.703696977694995951e+00,
9.411626586787267712e+00, 1.292994970279017330e+01,
1.511450835363295475e+01, 1.547505586325072358e+01,
1.674219613601154322e+01, 1.541441164390344731e+01,
2.321646656533689068e+01, 2.573145221936697169e+01,
2.863361908099808062e+01, 2.946321689971910729e+01,
4.116637683351552113e+01, 3.565156034608240532e+01,
2.875972875753828717e+01, 3.650824457755138752e+01,
3.147545791262981396e+01, 4.092292559068643243e+01,
4.351952493063727445e+01, 4.621054606476813831e+01,
4.899598899307903821e+01, 5.187585371556997416e+01,
5.485014023224093194e+01, 5.791884854309192576e+01,
6.108197864812294142e+01, 6.433953054733399313e+01,
6.769150424072508088e+01, 7.113789972829619046e+01,
7.467871701004733609e+01, 7.831395608597850355e+01,
8.204361695608972127e+01, 8.586769962038094661e+01,
8.978620407885220800e+01, 9.379913033150350543e+01,
9.790647837833483891e+01, 1.021082482193461942e+02,
1.064044398545375856e+02, 1.107950532839089988e+02,
1.152800885074604480e+02, 1.198595455251919333e+02,
1.245334243371034404e+02, 1.293017249431949836e+02,
1.341644473434665485e+02, 1.391215915379181354e+02,
1.441731575265497725e+02, 1.493191453093614314e+02,
1.545595548863531405e+02, 1.598943862575248431e+02,
1.653236394228765960e+02, 1.708473143824083991e+02,
1.764654111361201956e+02, 1.821779296840120423e+02,
1.879848700260839394e+02, 1.938862321623358298e+02,
                ])
elif which_string == "cello-a":
    T = 143.5
    L = 0.690
    d = 6.62e-4
    pl = 1.55e-3
    E = 4.74e9
    rn = numpy.array([
                1.313e+00, 1.006e+00, 1.981e+00, 1.661e+00,
                1.920e+00, 2.228e+00, 3.206e+00, 3.542e+00,
                4.542e+00, 5.236e+00, 5.631e+00, 6.810e+00,
                7.085e+00, 9.205e+00, 1.178e+01, 1.263e+01,
                1.529e+01, 1.922e+01, 2.358e+01, 2.498e+01,
                2.826e+01, 3.440e+01, 3.601e+01, 3.996e+01,
                4.379e+01, 5.244e+01, 5.870e+01, 6.546e+01,
                7.274e+01, 8.057e+01, 8.895e+01, 9.790e+01,
                1.075e+02, 1.176e+02, 1.284e+02, 1.399e+02,
                1.520e+02, 1.649e+02, 1.784e+02, 1.927e+02,
    ])
elif which_string == "cello-c":
    ### physical constants, cello C string
    T = 124.2        # tension, N
    L = 0.692       # length, m
    d = 1.65e-3      # diameter, m
    pl = 1.624e-2    # linear density, kg/m
    
    E = 4.0e9      # young's elastic modulus
    #rn = 7.8 + 0.4*(n) # extremely vague approximation of modal decays
    rn = numpy.array([
4.135236480243599799e-01, 5.770789200544175213e-01,
7.053602074433531488e-01, 6.588758047748748403e-01,
5.728509947877817865e-01, 9.845283268744289273e-01,
2.972222778616189220e+00, 2.291844232159464756e+00,
4.133075096220452771e+00, 5.369100840590236068e+00,
6.096347519059874820e+00, 7.051354175979470895e+00,
9.397315636881982570e+00, 1.249992562382186634e+01,
1.814259977128301671e+01, 2.173457512023855998e+01,
2.501585555686363449e+01, 3.404413419967075072e+01,
3.585774153291733057e+01, 3.341417460197163081e+01,
4.525122658581160096e+01, 5.222336179493000685e+01,
5.989215761970972807e+01, 6.829078837518225953e+01,
7.745242837637908906e+01, 8.741025193833171159e+01,
9.819743337607160072e+01, 1.098471470046302727e+02,
1.223925671390392154e+02, 1.358668680943299023e+02,
1.503032241855338498e+02, 1.657348097276825172e+02,
1.821947990358074208e+02, 1.997163664249400483e+02,
2.183326862101118593e+02, 2.380769327063543983e+02,
2.589822802286991532e+02, 2.810819030921775266e+02,
3.044089756118211199e+02, 3.289966721026613641e+02,
3.548781668797298039e+02, 3.820866342580578703e+02,
4.106552485526770511e+02, 4.406171840786188909e+02,
4.720056151509148208e+02, 5.048537160845963854e+02,
5.391946611946950725e+02, 5.750616247962423131e+02,
6.124877812042697087e+02, 6.515063047338086335e+02,
6.921503696998906889e+02, 7.344531504175472492e+02,
7.784478212018098020e+02, 8.241675563677100627e+02,
8.716455302302791779e+02, 9.209149171045489766e+02,
9.720088913055507192e+02, 1.024960627148315780e+03,
1.079803298947876101e+03, 1.136570081019262943e+03,
1.195294147677507681e+03, 1.256008673237641915e+03,
1.318746832014697020e+03, 1.383541798323704597e+03,
                ])
elif which_string == "unit":
    T = 400.0        # tension, N
    ### test with B5
    L = 2.0       # length, m
    #pl = 10.0e-4    # linear density, kg/m
    pl = 1e-4    # linear density, kg/m
    
    d = 0      # diameter, m
    E = 0.0             # let's try an ideal string
    rn = numpy.zeros(N) # no damping at all for now
else:
    print "**** invalid string type!"
    exit(1)
    

### friction characteristics
mu_s = 0.8
mu_d = 0.3
mu_v0 = 0.1
