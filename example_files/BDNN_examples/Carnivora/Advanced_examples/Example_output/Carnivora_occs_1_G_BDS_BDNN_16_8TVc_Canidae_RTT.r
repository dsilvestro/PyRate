pdf(file='/home/torsten/Work/Software/PyRate/example_files/BDNN_examples/Carnivora/Advanced_examples/Example_output/Carnivora_occs_1_G_BDS_BDNN_16_8TVc_Canidae_RTT.pdf', width = 9, height = 6, useDingbats = FALSE)

layout(matrix(1:4, ncol = 2, nrow = 2, byrow = TRUE))
par(las = 1, mar = c(4.5, 4.5, 0.5, 0.5))
time_vec=c(23.670092814507477, 22.00001,22.0,21.00001,21.0,20.00001,20.0,19.00001,19.0,18.00001,18.0,17.00001,17.0,16.00001,16.0,15.00001,15.0,14.00001,14.0,13.00001,13.0,12.00001,12.0,11.00001,11.0,10.00001,10.0,9.00001,9.0,8.00001,8.0,7.00001,7.0,6.00001,6.0,5.00001,5.0,4.00001,4.0,3.00001,3.0,2.00001,2.0,1.00001,1.0,0.0)
sp_mean=c(0.4980512672733635, 0.4980512672733635,0.38934964540688854,0.38934964540688854,0.3090618553296267,0.3090618553296267,0.250065635616937,0.250065635616937,0.1702503810217736,0.1702503810217736,0.12361523519332342,0.12361523519332342,0.09166938469073012,0.09166938469073012,0.07471567948620482,0.07471567948620482,0.06339960757746545,0.06339960757746545,0.0632950138473455,0.0632950138473455,0.06564442857372277,0.06564442857372277,0.062020282532722126,0.062020282532722126,0.05568769921343087,0.05568769921343087,0.06385053839360769,0.06385053839360769,0.07607476415531891,0.07607476415531891,0.0651322281856434,0.0651322281856434,0.07205022714682609,0.07205022714682609,0.080995624767907,0.080995624767907,0.10168056010911339,0.10168056010911339,0.12031528126952383,0.12031528126952383,0.1488599600482648,0.1488599600482648,0.16153273145393235,0.16153273145393235,0.16765470472801522,0.16765470472801522)
sp_lwr=c(0.21828301920252272, 0.21828301920252272,0.20491643137043591,0.20491643137043591,0.15653023286127102,0.15653023286127102,0.1217026857950428,0.1217026857950428,0.08840523083153079,0.08840523083153079,0.052586628991021504,0.052586628991021504,0.033129106382482516,0.033129106382482516,0.03156338398871539,0.03156338398871539,0.013024715991735195,0.013024715991735195,0.024063476438084187,0.024063476438084187,0.014634935865888297,0.014634935865888297,0.022617969255929807,0.022617969255929807,0.01412919187530756,0.01412919187530756,0.023297046743024262,0.023297046743024262,0.033867471376804105,0.033867471376804105,0.015301212502142177,0.015301212502142177,0.015495188160257894,0.015495188160257894,0.028626190189102976,0.028626190189102976,0.02958732464010201,0.02958732464010201,0.039028200386175126,0.039028200386175126,0.061192148928060276,0.061192148928060276,0.05500489367091733,0.05500489367091733,0.05975771636212862,0.05975771636212862)
sp_upr=c(0.810354483917672, 0.810354483917672,0.5978149189520078,0.5978149189520078,0.46571054338968093,0.46571054338968093,0.39328826704511755,0.39328826704511755,0.25465501606417984,0.25465501606417984,0.1911190041411443,0.1911190041411443,0.185683614918945,0.185683614918945,0.1357936793828712,0.1357936793828712,0.10995093847403581,0.10995093847403581,0.11101464866431889,0.11101464866431889,0.10584213556051096,0.10584213556051096,0.10378269515129247,0.10378269515129247,0.09293032470078383,0.09293032470078383,0.10394535628222631,0.10394535628222631,0.14155937378721228,0.14155937378721228,0.11256570860562315,0.11256570860562315,0.12253240195578288,0.12253240195578288,0.1576145789049194,0.1576145789049194,0.18570584338611903,0.18570584338611903,0.21491677080772828,0.21491677080772828,0.2703851053113247,0.2703851053113247,0.2719376313052549,0.2719376313052549,0.30714686779904177,0.30714686779904177)
ex_mean=c(0.08414383372507017, 0.08414383372507017,0.08595869598903554,0.08595869598903554,0.08681639922259742,0.08681639922259742,0.08309789291647479,0.08309789291647479,0.08571687117398218,0.08571687117398218,0.08931199674824092,0.08931199674824092,0.10595668362818658,0.10595668362818658,0.09697379320311142,0.09697379320311142,0.09382847247209808,0.09382847247209808,0.08525352189600426,0.08525352189600426,0.10185584285480029,0.10185584285480029,0.09608948502258477,0.09608948502258477,0.09229810796802071,0.09229810796802071,0.1054064173595027,0.1054064173595027,0.11395721449897979,0.11395721449897979,0.10620498572564176,0.10620498572564176,0.10133046481449977,0.10133046481449977,0.10313667344865964,0.10313667344865964,0.11363418102475607,0.11363418102475607,0.12612834709388948,0.12612834709388948,0.217638305398718,0.217638305398718,0.39081544671987356,0.39081544671987356,0.8954039225836656,0.8954039225836656)
ex_lwr=c(0.03172909081225223, 0.03172909081225223,0.031757773440912185,0.031757773440912185,0.043271716080370566,0.043271716080370566,0.0335048396747079,0.0335048396747079,0.04265079217013346,0.04265079217013346,0.04371503432775972,0.04371503432775972,0.041838082667707936,0.041838082667707936,0.04142572250869985,0.04142572250869985,0.04369623573044641,0.04369623573044641,0.0341421808267758,0.0341421808267758,0.06133010054319734,0.06133010054319734,0.050126728330571586,0.050126728330571586,0.051130612567453146,0.051130612567453146,0.04880556453639158,0.04880556453639158,0.04643513306334309,0.04643513306334309,0.042576935763618734,0.042576935763618734,0.03380520343679749,0.03380520343679749,0.030735258117752007,0.030735258117752007,0.04771362068272305,0.04771362068272305,0.06093418670463216,0.06093418670463216,0.07005580260084449,0.07005580260084449,0.14504597625143284,0.14504597625143284,0.5128223281161194,0.5128223281161194)
ex_upr=c(0.14174459322413474, 0.14174459322413474,0.13365671118321798,0.13365671118321798,0.13375557388775222,0.13375557388775222,0.13188262238060994,0.13188262238060994,0.13404756193647888,0.13404756193647888,0.13363387968884866,0.13363387968884866,0.181333472500679,0.181333472500679,0.14291603324516552,0.14291603324516552,0.14310343320961139,0.14310343320961139,0.12817302546821405,0.12817302546821405,0.16779311705007585,0.16779311705007585,0.14653958966192232,0.14653958966192232,0.1408070612350552,0.1408070612350552,0.15445448461868239,0.15445448461868239,0.1699060076017681,0.1699060076017681,0.16005496815402884,0.16005496815402884,0.15933001815739273,0.15933001815739273,0.16779519844388877,0.16779519844388877,0.1954848731086804,0.1954848731086804,0.21824181040497848,0.21824181040497848,0.3595892613396403,0.3595892613396403,0.6491661490135464,0.6491661490135464,1.4187139562401243,1.4187139562401243)
div_mean=c(0.4139074335482935, 0.4139074335482935,0.303390949417853,0.303390949417853,0.22224545610702895,0.22224545610702895,0.16696774270046225,0.16696774270046225,0.08453350984779133,0.08453350984779133,0.03430323844508255,0.03430323844508255,-0.014287298937456484,-0.014287298937456484,-0.022258113716906603,-0.022258113716906603,-0.030428864894632625,-0.030428864894632625,-0.021958508048658704,-0.021958508048658704,-0.03621141428107752,-0.03621141428107752,-0.034069202489862616,-0.034069202489862616,-0.03661040875458991,-0.03661040875458991,-0.04155587896589508,-0.04155587896589508,-0.0378824503436609,-0.0378824503436609,-0.04107275753999842,-0.04107275753999842,-0.029280237667673698,-0.029280237667673698,-0.02214104868075268,-0.02214104868075268,-0.011953620915642655,-0.011953620915642655,-0.005813065824365588,-0.005813065824365588,-0.06877834535045331,-0.06877834535045331,-0.22928271526594118,-0.22928271526594118,-0.72774921785565,-0.72774921785565)
div_lwr=c(0.07653842597838798, 0.07653842597838798,0.08592711165104985,0.08592711165104985,0.069650800609675,0.069650800609675,0.04267913042021221,0.04267913042021221,-0.01835037563668787,-0.01835037563668787,-0.043178846783961736,-0.043178846783961736,-0.114163099033402,-0.114163099033402,-0.09989100763513492,-0.09989100763513492,-0.10351923126823667,-0.10351923126823667,-0.08508507121594344,-0.08508507121594344,-0.09980409952857311,-0.09980409952857311,-0.08518666716520865,-0.08518666716520865,-0.08979568367019171,-0.08979568367019171,-0.11043773067068798,-0.11043773067068798,-0.12725786865444277,-0.12725786865444277,-0.11119607056913101,-0.11119607056913101,-0.12239044944260616,-0.12239044944260616,-0.11489027696784843,-0.11489027696784843,-0.1312860209713867,-0.1312860209713867,-0.16403495640694798,-0.16403495640694798,-0.24619180991845127,-0.24619180991845127,-0.5170383976017101,-0.5170383976017101,-1.267227188029378,-1.267227188029378)
div_upr=c(0.7074270937087407, 0.7074270937087407,0.49655504281443064,0.49655504281443064,0.3964969213734522,0.3964969213734522,0.33409796551860255,0.33409796551860255,0.18346710228869761,0.18346710228869761,0.1309685150437503,0.1309685150437503,0.11606266117237134,0.11606266117237134,0.06103373761032456,0.06103373761032456,0.04212327759330026,0.04212327759330026,0.06114630766124466,0.06114630766124466,0.04797715715609654,0.04797715715609654,0.044820915651041085,0.044820915651041085,0.03347788163761918,0.03347788163761918,0.029340404225257105,0.029340404225257105,0.030933184532957944,0.030933184532957944,0.03456846082747159,0.03456846082747159,0.05005962535111763,0.05005962535111763,0.08187202090572904,0.08187202090572904,0.09506541076886209,0.09506541076886209,0.1019397678729186,0.1019397678729186,0.1304621839699861,0.1304621839699861,0.06861135053052664,0.06861135053052664,-0.34738707000093405,-0.34738707000093405)
long_mean=c(13.650032072660384, 13.650032072660384,12.934645182581562,12.934645182581562,12.67059889636777,12.67059889636777,13.491115741848311,13.491115741848311,12.749752592478616,12.749752592478616,12.142321382505349,12.142321382505349,10.717021922328275,10.717021922328275,11.36414340044723,11.36414340044723,11.783240277767662,11.783240277767662,12.984753358931414,12.984753358931414,10.60931442615557,10.60931442615557,11.230704121351668,11.230704121351668,11.76239095192925,11.76239095192925,10.355541361048367,10.355541361048367,9.676117823123356,9.676117823123356,10.373096273412667,10.373096273412667,11.337992493365457,11.337992493365457,11.31444542433755,11.31444542433755,10.178328071551071,10.178328071551071,9.36684820497956,9.36684820497956,5.458812715619075,5.458812715619075,2.9639423417216366,2.9639423417216366,1.2177085360031457,1.2177085360031457)
long_lwr=c(5.932751129056513, 5.932751129056513,6.062996382542056,6.062996382542056,6.5481022209295805,6.5481022209295805,6.440370660450255,6.440370660450255,6.706362068183847,6.706362068183847,6.805392922230396,6.805392922230396,4.265942577481325,4.265942577481325,5.0509084167739005,5.0509084167739005,6.797569053943659,6.797569053943659,6.932507866423145,6.932507866423145,5.959720026546512,5.959720026546512,6.333136885939848,6.333136885939848,6.220621131272661,6.220621131272661,4.306949272956944,4.306949272956944,4.055711601096693,4.055711601096693,4.548122367649352,4.548122367649352,4.548835741251215,4.548835741251215,4.452763112163291,4.452763112163291,4.2590908218643495,4.2590908218643495,4.462789688922513,4.462789688922513,2.441698651477081,2.441698651477081,1.2755402355652115,1.2755402355652115,0.6537434516395768,0.6537434516395768)
long_upr=c(23.942909826131203, 23.942909826131203,21.036994608602207,21.036994608602207,19.778580288792877,19.778580288792877,22.178389321022028,22.178389321022028,20.51354929196087,20.51354929196087,18.893514519974683,18.893514519974683,18.532567545021834,18.532567545021834,19.11835589554523,19.11835589554523,21.303876614700616,21.303876614700616,22.250616680975533,22.250616680975533,16.30520724967112,16.30520724967112,17.706303006802408,17.706303006802408,18.31705627103307,18.31705627103307,16.081574296846593,16.081574296846593,15.16113055573635,15.16113055573635,17.645486259652266,17.645486259652266,22.482786393858266,22.482786393858266,21.537770461758743,21.537770461758743,17.05848717549098,17.05848717549098,15.893983518580617,15.893983518580617,11.152280425054935,11.152280425054935,5.696630305678007,5.696630305678007,1.812472194187685,1.812472194187685)
xlim = c(23.670092814507477, 0.0)
ylim = c(0.013024715991735195, 0.810354483917672)
not_NA = !is.na(sp_mean)
plot(time_vec[not_NA], sp_mean[not_NA], type = 'n', xlim = xlim, ylim = ylim, xlab = 'Time (Ma)', ylab = 'Speciation rate')
polygon(c(time_vec[not_NA], rev(time_vec[not_NA])), c(sp_lwr[not_NA], rev(sp_upr[not_NA])), col = adjustcolor('#4c4cec', alpha = 0.5), border = NA)
lines(time_vec[not_NA], sp_mean[not_NA], col = '#4c4cec', lwd = 2)
ylim = c(0.030735258117752007, 1.4187139562401243)
not_NA = !is.na(ex_mean)
plot(time_vec[not_NA], ex_mean[not_NA], type = 'n', xlim = xlim, ylim = ylim, xlab = 'Time (Ma)', ylab = 'Extinction rate')
polygon(c(time_vec[not_NA], rev(time_vec[not_NA])), c(ex_lwr[not_NA], rev(ex_upr[not_NA])), col = adjustcolor('#e34a33', alpha = 0.5), border = NA)
lines(time_vec[not_NA], ex_mean[not_NA], col = '#e34a33', lwd = 2)
ylim = c(-1.267227188029378, 0.7074270937087407)
not_NA = !is.na(div_mean)
plot(time_vec[not_NA], div_mean[not_NA], type = 'n', xlim = xlim, ylim = ylim, xlab = 'Time (Ma)', ylab = 'Net diversification rate')
polygon(c(time_vec[not_NA], rev(time_vec[not_NA])), c(div_lwr[not_NA], rev(div_upr[not_NA])), col = adjustcolor('black', alpha = 0.3), border = NA)
lines(time_vec[not_NA], div_mean[not_NA], col = 'black', lwd = 2)
abline(h = 0, col = 'red', lty = 2)
ylim = c(0.6537434516395768, 23.942909826131203)
not_NA = !is.na(long_mean)
plot(time_vec[not_NA], long_mean[not_NA], type = 'n', xlim = xlim, ylim = ylim, xlab = 'Time (Ma)', ylab = 'Longevity (Myr)')
polygon(c(time_vec[not_NA], rev(time_vec[not_NA])), c(long_lwr[not_NA], rev(long_upr[not_NA])), col = adjustcolor('black', alpha = 0.3), border = NA)
lines(time_vec[not_NA], long_mean[not_NA], col = 'black', lwd = 2)
dev.off()