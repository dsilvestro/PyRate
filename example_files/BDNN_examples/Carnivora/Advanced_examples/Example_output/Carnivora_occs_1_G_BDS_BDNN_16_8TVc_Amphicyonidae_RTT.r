pdf(file='/home/torsten/Work/Software/PyRate/example_files/BDNN_examples/Carnivora/Advanced_examples/Example_output/Carnivora_occs_1_G_BDS_BDNN_16_8TVc_Amphicyonidae_RTT.pdf', width = 9, height = 6, useDingbats = FALSE)

layout(matrix(1:4, ncol = 2, nrow = 2, byrow = TRUE))
par(las = 1, mar = c(4.5, 4.5, 0.5, 0.5))
time_vec=c(23.670092814507477, 22.00001,22.0,21.00001,21.0,20.00001,20.0,19.00001,19.0,18.00001,18.0,17.00001,17.0,16.00001,16.0,15.00001,15.0,14.00001,14.0,13.00001,13.0,12.00001,12.0,11.00001,11.0,10.00001,10.0,9.00001,9.0,8.00001,8.0,7.00001,7.0,6.00001,6.0,5.00001,5.0,4.00001,4.0,3.00001,3.0,2.00001,2.0,1.00001,1.0,0.0)
sp_mean=c(0.35033772433655647, 0.35033772433655647,0.29037256242396825,0.29037256242396825,0.2302752283983756,0.2302752283983756,0.19739580741263718,0.19739580741263718,0.1319445777691946,0.1319445777691946,0.09629861466702684,0.09629861466702684,0.08971227043625093,0.08971227043625093,0.07471267819369756,0.07471267819369756,0.06355385172963146,0.06355385172963146,0.06355421114018799,0.06355421114018799,0.06671984416724247,0.06671984416724247,0.05987816908276937,0.05987816908276937,0.0515655989595077,0.0515655989595077,0.050454791772612764,0.050454791772612764,0.0507692686432567,0.0507692686432567,0.04190173714764808,0.04190173714764808,0.04158648385949616,0.04158648385949616,0.04129213645008419,0.04129213645008419,0.04091871498767224,0.04091871498767224,0.04271948155231105,0.04271948155231105,0.04860525519306324,0.04860525519306324,NA,NA,NA,NA)
sp_lwr=c(0.12038036949847306, 0.12038036949847306,0.1107832670139513,0.1107832670139513,0.09939527978075591,0.09939527978075591,0.08409171084261267,0.08409171084261267,0.05362124017802557,0.05362124017802557,0.029625154475793853,0.029625154475793853,0.025318479894107456,0.025318479894107456,0.02268214494437742,0.02268214494437742,0.018191492680393037,0.018191492680393037,0.021363355837871953,0.021363355837871953,0.021565714686875997,0.021565714686875997,0.013143529875959134,0.013143529875959134,0.011749242641406251,0.011749242641406251,0.009023785999005773,0.009023785999005773,0.004703726837944754,0.004703726837944754,0.00418652570179777,0.00418652570179777,0.004717459103830976,0.004717459103830976,0.003524872152294049,0.003524872152294049,0.0022640380099043355,0.0022640380099043355,0.002617110302805709,0.002617110302805709,0.0054209465658918675,0.0054209465658918675,NA,NA,NA,NA)
sp_upr=c(0.5634635367812959, 0.5634635367812959,0.4656553834329694,0.4656553834329694,0.4042232092904036,0.4042232092904036,0.32224429417630557,0.32224429417630557,0.23405093313870737,0.23405093313870737,0.16650794752747639,0.16650794752747639,0.17910075058969804,0.17910075058969804,0.1377481670101852,0.1377481670101852,0.12106892498347795,0.12106892498347795,0.1205800317946093,0.1205800317946093,0.13193239852667338,0.13193239852667338,0.10364835308402165,0.10364835308402165,0.09631413148736682,0.09631413148736682,0.0974811520005037,0.0974811520005037,0.0968050807104695,0.0968050807104695,0.0835982261222496,0.0835982261222496,0.08869563968576,0.08869563968576,0.09059214321780501,0.09059214321780501,0.09279537483931716,0.09279537483931716,0.09724705690976722,0.09724705690976722,0.10054866854050154,0.10054866854050154,NA,NA,NA,NA)
ex_mean=c(0.0872894992841078, 0.0872894992841078,0.08743324898037882,0.08743324898037882,0.08692826378000634,0.08692826378000634,0.10389374445623833,0.10389374445623833,0.08517299665628092,0.08517299665628092,0.07765458872388761,0.07765458872388761,0.07850394565368331,0.07850394565368331,0.08119828819811177,0.08119828819811177,0.08277360264257379,0.08277360264257379,0.1104426277540432,0.1104426277540432,0.14280465238276455,0.14280465238276455,0.1470732802864498,0.1470732802864498,0.14302828081108027,0.14302828081108027,0.1738662490567709,0.1738662490567709,0.19885262786455438,0.19885262786455438,0.1604136909557105,0.1604136909557105,0.1782734537796442,0.1782734537796442,0.1974119969600592,0.1974119969600592,0.22321180364267487,0.22321180364267487,0.3137972789624486,0.3137972789624486,0.7581910875585708,0.7581910875585708,NA,NA,NA,NA)
ex_lwr=c(0.031871044206640074, 0.031871044206640074,0.04012211871615607,0.04012211871615607,0.04125942640584022,0.04125942640584022,0.04178186516296896,0.04178186516296896,0.036737976509599056,0.036737976509599056,0.03089915968862931,0.03089915968862931,0.024148837645905,0.024148837645905,0.027505712379009815,0.027505712379009815,0.02628466614360326,0.02628466614360326,0.051654281621138456,0.051654281621138456,0.054002136234642686,0.054002136234642686,0.07548978904757031,0.07548978904757031,0.0645273285553358,0.0645273285553358,0.07095733366309682,0.07095733366309682,0.043185055751204154,0.043185055751204154,0.04437131427890121,0.04437131427890121,0.05097887767483834,0.05097887767483834,0.03525931440377403,0.03525931440377403,0.03287630048481727,0.03287630048481727,0.03712413909731645,0.03712413909731645,0.10036099403668461,0.10036099403668461,NA,NA,NA,NA)
ex_upr=c(0.14442527922282308, 0.14442527922282308,0.14416562300692404,0.14416562300692404,0.14079373004900247,0.14079373004900247,0.16052061970267592,0.16052061970267592,0.1348250667793068,0.1348250667793068,0.13063579531406339,0.13063579531406339,0.13388725407142796,0.13388725407142796,0.131240551289695,0.131240551289695,0.13627498809287275,0.13627498809287275,0.18807515384113413,0.18807515384113413,0.23457702228992713,0.23457702228992713,0.25714011138826853,0.25714011138826853,0.24709897476942497,0.24709897476942497,0.298671633484052,0.298671633484052,0.3625275677643508,0.3625275677643508,0.3075851573743553,0.3075851573743553,0.35946004570609064,0.35946004570609064,0.4187996249362245,0.4187996249362245,0.5286002301900425,0.5286002301900425,0.7673031951494302,0.7673031951494302,1.4813524287883988,1.4813524287883988,NA,NA,NA,NA)
div_mean=c(0.26304822505244857, 0.26304822505244857,0.20293931344358948,0.20293931344358948,0.14334696461836924,0.14334696461836924,0.0935020629563988,0.0935020629563988,0.046771581112913733,0.046771581112913733,0.01864402594313916,0.01864402594313916,0.011208324782567593,0.011208324782567593,-0.006485610004414158,-0.006485610004414158,-0.019219750912942292,-0.019219750912942292,-0.04688841661385522,-0.04688841661385522,-0.07608480821552212,-0.07608480821552212,-0.0871951112036805,-0.0871951112036805,-0.09146268185157258,-0.09146268185157258,-0.12341145728415828,-0.12341145728415828,-0.14808335922129767,-0.14808335922129767,-0.1185119538080624,-0.1185119538080624,-0.1366869699201479,-0.1366869699201479,-0.15611986050997498,-0.15611986050997498,-0.18229308865500252,-0.18229308865500252,-0.27107779741013766,-0.27107779741013766,-0.7095858323655069,-0.7095858323655069,NA,NA,NA,NA)
div_lwr=c(0.05511641031298767, 0.05511641031298767,0.01716514066276291,0.01716514066276291,0.03281392383153292,0.03281392383153292,-0.06099290801740068,-0.06099290801740068,-0.046535271600020314,-0.046535271600020314,-0.05369869479030459,-0.05369869479030459,-0.061922853831703775,-0.061922853831703775,-0.07461845601232672,-0.07461845601232672,-0.08123649357397333,-0.08123649357397333,-0.13692775826398218,-0.13692775826398218,-0.1968927453909925,-0.1968927453909925,-0.2107320356230954,-0.2107320356230954,-0.20208396061163683,-0.20208396061163683,-0.2595993040579851,-0.2595993040579851,-0.3456055819579822,-0.3456055819579822,-0.2728032260095469,-0.2728032260095469,-0.3318476752047542,-0.3318476752047542,-0.4165560679063778,-0.4165560679063778,-0.4898899950636591,-0.4898899950636591,-0.7735159429190064,-0.7735159429190064,-1.410239644980434,-1.410239644980434,NA,NA,NA,NA)
div_upr=c(0.504102836266524, 0.504102836266524,0.38778508236540027,0.38778508236540027,0.335730425669034,0.335730425669034,0.21816891160779964,0.21816891160779964,0.1802271551833774,0.1802271551833774,0.11982962206644784,0.11982962206644784,0.14164493815696974,0.14164493815696974,0.08463374615245112,0.08463374615245112,0.06860055741979376,0.06860055741979376,0.05565771707493058,0.05565771707493058,0.018062995030094883,0.018062995030094883,0.004892466517434399,0.004892466517434399,0.004011999976193989,0.004011999976193989,-0.00942524772162387,-0.00942524772162387,-0.005793250280036855,-0.005793250280036855,0.005596012460417063,0.005596012460417063,-0.00319872024235289,-0.00319872024235289,-0.00510662169384888,-0.00510662169384888,0.05423230349939449,0.05423230349939449,0.008247909152993331,0.008247909152993331,-0.030877386633641216,-0.030877386633641216,NA,NA,NA,NA)
long_mean=c(13.10986003058213, 13.10986003058213,12.897623690991878,12.897623690991878,12.93227561024045,12.93227561024045,10.783701869164563,10.783701869164563,13.280703632195666,13.280703632195666,14.707957225974907,14.707957225974907,15.034766803593731,15.034766803593731,14.077177351153903,14.077177351153903,13.82136472731078,13.82136472731078,10.219109792879674,10.219109792879674,8.003197529199497,8.003197529199497,7.773101769994872,7.773101769994872,8.033177165610093,8.033177165610093,6.675865621700041,6.675865621700041,6.2135159009278915,6.2135159009278915,7.9056281339506835,7.9056281339506835,7.301958345690508,7.301958345690508,6.964444545361164,6.964444545361164,7.1020065486762665,7.1020065486762665,5.432633170399991,5.432633170399991,1.9897707334355457,1.9897707334355457,NA,NA,NA,NA)
long_lwr=c(4.65976004533795, 4.65976004533795,6.476648547577829,6.476648547577829,4.961172029301999,4.961172029301999,4.906164479715263,4.906164479715263,6.985984588851774,6.985984588851774,5.1371133891693,5.1371133891693,5.490443018976691,5.490443018976691,6.0817156195417486,6.0817156195417486,6.312477205163763,6.312477205163763,4.011461808540805,4.011461808540805,3.6708533839308353,3.6708533839308353,3.888930414633175,3.888930414633175,3.6823284483889163,3.6823284483889163,2.7255585048923763,2.7255585048923763,2.2390899509859064,2.2390899509859064,2.1007572946641733,2.1007572946641733,2.3639350422021383,2.3639350422021383,1.5148096835103304,1.5148096835103304,1.2447621008435152,1.2447621008435152,0.7169246532272904,0.7169246532272904,0.5000316415715971,0.5000316415715971,NA,NA,NA,NA)
long_upr=c(22.916694525403255, 22.916694525403255,22.81566950246353,22.81566950246353,21.239582764370052,21.239582764370052,18.883362826114553,18.883362826114553,24.83308010649686,24.83308010649686,23.873055356053023,23.873055356053023,25.944761405239493,25.944761405239493,24.14466414239889,24.14466414239889,23.629437639879107,23.629437639879107,16.44396652322869,16.44396652322869,13.779134899444974,13.779134899444974,13.24682467147768,13.24682467147768,14.135906147676208,14.135906147676208,11.328523793778341,11.328523793778341,12.010246984778213,12.010246984778213,14.13656085111491,14.13656085111491,14.555953109615638,14.555953109615638,14.378166463106616,14.378166463106616,16.27264477839274,16.27264477839274,15.162445247261655,15.162445247261655,4.040164873224015,4.040164873224015,NA,NA,NA,NA)
xlim = c(23.670092814507477, 0.0)
ylim = c(0.0022640380099043355, 0.5634635367812959)
not_NA = !is.na(sp_mean)
plot(time_vec[not_NA], sp_mean[not_NA], type = 'n', xlim = xlim, ylim = ylim, xlab = 'Time (Ma)', ylab = 'Speciation rate')
polygon(c(time_vec[not_NA], rev(time_vec[not_NA])), c(sp_lwr[not_NA], rev(sp_upr[not_NA])), col = adjustcolor('#4c4cec', alpha = 0.5), border = NA)
lines(time_vec[not_NA], sp_mean[not_NA], col = '#4c4cec', lwd = 2)
ylim = c(0.024148837645905, 1.4813524287883988)
not_NA = !is.na(ex_mean)
plot(time_vec[not_NA], ex_mean[not_NA], type = 'n', xlim = xlim, ylim = ylim, xlab = 'Time (Ma)', ylab = 'Extinction rate')
polygon(c(time_vec[not_NA], rev(time_vec[not_NA])), c(ex_lwr[not_NA], rev(ex_upr[not_NA])), col = adjustcolor('#e34a33', alpha = 0.5), border = NA)
lines(time_vec[not_NA], ex_mean[not_NA], col = '#e34a33', lwd = 2)
ylim = c(-1.410239644980434, 0.504102836266524)
not_NA = !is.na(div_mean)
plot(time_vec[not_NA], div_mean[not_NA], type = 'n', xlim = xlim, ylim = ylim, xlab = 'Time (Ma)', ylab = 'Net diversification rate')
polygon(c(time_vec[not_NA], rev(time_vec[not_NA])), c(div_lwr[not_NA], rev(div_upr[not_NA])), col = adjustcolor('black', alpha = 0.3), border = NA)
lines(time_vec[not_NA], div_mean[not_NA], col = 'black', lwd = 2)
abline(h = 0, col = 'red', lty = 2)
ylim = c(0.5000316415715971, 25.944761405239493)
not_NA = !is.na(long_mean)
plot(time_vec[not_NA], long_mean[not_NA], type = 'n', xlim = xlim, ylim = ylim, xlab = 'Time (Ma)', ylab = 'Longevity (Myr)')
polygon(c(time_vec[not_NA], rev(time_vec[not_NA])), c(long_lwr[not_NA], rev(long_upr[not_NA])), col = adjustcolor('black', alpha = 0.3), border = NA)
lines(time_vec[not_NA], long_mean[not_NA], col = 'black', lwd = 2)
dev.off()