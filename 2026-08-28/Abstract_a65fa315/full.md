# A<sub>es</sub>C<sub>a</sub>n<sub>vas:</sub> A L<sub>a</sub>r<sub>ge-</sub>S<sub>ca</sub>l<sub>e</sub> D<sub>a</sub>t<sub>ase</sub>t <sub>a</sub>nd B<sub>e</sub>n<sub>c</sub>hm<sub>a</sub>rk f<sub>o</sub>r A<sub>es</sub>th<sub>e</sub>ti<sub>c</sub> Criti<sub>que a</sub>nd C<sub>o</sub>nt<sub>ex</sub>t<sub>ua</sub>l S<sub>u</sub>it<sub>a</sub>bilit<sub>y</sub>

<sub>Xuanwei</sub> <sub>Hu</sub>1∗<sub>,</sub> <sub>Haoyu</sub> <sub>Dong</sub>1∗<sub>,</sub> <sub>Kejun</sub> <sub>Wu</sub>1†<sub>,</sub> <sub>Tianyi</sub> <sub>Liu</sub>2<sub>,</sub> <sub>Jianjun</sub> <sub>Gao</sub>2

<sup>1</sup>S<sub>c</sub>h<sub>oo</sub>l <sub>o</sub>f El<sub>ec</sub>t<sub>ron</sub>i<sub>c</sub> I<sub>n</sub>f<sub>orma</sub>ti<sub>on an</sub>d C<sub>ommun</sub>i<sub>ca</sub>ti<sub>ons,</sub> H<sub>uaz</sub>h<sub>ong</sub> U<sub>n</sub>i<sub>vers</sub>it<sub>y o</sub>f S<sub>c</sub>i<sub>ence an</sub>d T<sub>ec</sub>h<sub>no</sub>l<sub>ogy,</sub> W<sub>u</sub>h<sub>an,</sub> Chi<sub>na</sub> <sup>2</sup>S<sub>c</sub>h<sub>oo</sub>l <sub>o</sub>f El<sub>ec</sub>t<sub>r</sub>i<sub>ca</sub>l <sub>an</sub>d El<sub>ec</sub>t<sub>ron</sub>i<sub>c</sub> E<sub>ng</sub>i<sub>neer</sub>i<sub>ng,</sub> N<sub>anyang</sub> T<sub>ec</sub>h<sub>no</sub>l<sub>og</sub>i<sub>ca</sub>l U<sub>n</sub>i<sub>vers</sub>it<sub>y,</sub> Si<sub>ngapore</sub> kjwu@hust.edu.cn

## Abstract

R<sub>ecen</sub>t <sub>a</sub>d<sub>vances</sub> i<sub>n</sub> M<sub>u</sub>lti<sub>mo</sub>d<sub>a</sub>l L<sub>arge</sub> L<sub>anguage</sub> M<sub>o</sub>d<sub>e</sub>l<sub>s</sub> (MLLMs) have extended Ima<sub>g</sub>e Aesthetic Assessment (IAA) b<sub>eyon</sub>d <sub>sca</sub>l<sub>ar scores</sub> t<sub>owar</sub>d i<sub>n</sub>t<sub>erpre</sub>t<sub>a</sub>bl<sub>e cr</sub>iti<sub>que an</sub>d <sub>gu</sub>id<sub>-</sub> <sub>ance.</sub> Y<sub>e</sub>t <sub>ex</sub>i<sub>s</sub>ti<sub>ng</sub> b<sub>enc</sub>h<sub>mar</sub>k<sub>s ma</sub>i<sub>n</sub>l<sub>y assess</sub> i<sub>n</sub>t<sub>r</sub>i<sub>ns</sub>i<sub>c v</sub>i<sub>sua</sub>l <sub>qua</sub>lit<sub>y</sub> <sub>or</sub> fi<sub>xe</sub>d d<sub>oma</sub>i<sub>n</sub> <sub>cr</sub>it<sub>er</sub>i<sub>a,</sub> l<sub>eav</sub>i<sub>ng</sub> <sub>open</sub> <sub>w</sub>h<sub>e</sub>th<sub>er</sub> <sub>an</sub> <sub>ap-</sub> <sub>pea</sub>li<sub>ng</sub> i<sub>mage</sub> i<sub>s appropr</sub>i<sub>a</sub>t<sub>e</sub> f<sub>or a spec</sub>ifi<sub>c purpose, au</sub>di<sub>ence,</sub> <sub>cu</sub>lt<sub>ura</sub>l <sub>se</sub>tti<sub>ng, or</sub> d<sub>oma</sub>i<sub>n conven</sub>ti<sub>on.</sub> W<sub>e</sub> i<sub>n</sub>t<sub>ro</sub>d<sub>uce</sub> A<sub>es</sub>C<sub>an-</sub> <sub>vas, a un</sub>ifi<sub>e</sub>d <sub>su</sub>it<sub>e w</sub>ith t<sub>wo comp</sub>l<sub>emen</sub>t<sub>ary componen</sub>t<sub>s:</sub> C<sub>ritique</sub>C<sub>anvas</sub> <sub>w</sub>ith 519<sub>,</sub>136 i<sub>ns</sub>t<sub>ruc</sub>ti<sub>on–response</sub> <sub>pa</sub>i<sub>rs</sub> f<sub>rom</sub> 54<sub>,</sub>300 i<sub>mages suppor</sub>t<sub>s</sub> l<sub>ong-</sub>f<sub>orm, mu</sub>lti<sub>-</sub>di<sub>mens</sub>i<sub>ona</sub>l <sub>cr</sub>iti<sub>que across p</sub>h<sub>o</sub>t<sub>ograp</sub>h<sub>y, pa</sub>i<sub>n</sub>ti<sub>ng, an</sub>d <sub>v</sub>i<sub>r</sub>t<sub>ua</sub>l i<sub>magery,</sub> <sub>w</sub>h<sub>ereas</sub> C<sub>ontext</sub>C<sub>anvas w</sub>ith 301 <sub>exper</sub>t<sub>-rev</sub>i<sub>ewe</sub>d <sub>use sce-</sub> <sub>nar</sub>i<sub>os eva</sub>l<sub>ua</sub>t<sub>es con</sub>t<sub>ex</sub>t<sub>ua</sub>l <sub>aes</sub>th<sub>e</sub>ti<sub>c su</sub>it<sub>a</sub>bilit<sub>y</sub> i<sub>n rea</sub>li<sub>s</sub>ti<sub>c use</sub> <sub>scenar</sub>i<sub>os.</sub> U<sub>n</sub>d<sub>er a un</sub>ifi<sub>e</sub>d <sub>pro</sub>t<sub>oco</sub>l<sub>, we eva</sub>l<sub>ua</sub>t<sub>e c</sub>l<sub>ose</sub>d<sub>-source</sub> f<sub>ron</sub>ti<sub>er, open-we</sub>i<sub>g</sub>ht <sub>genera</sub>l<sub>, an</sub>d <sub>aes</sub>th<sub>e</sub>ti<sub>c-spec</sub>ifi<sub>c</sub> MLLM<sub>s.</sub> R<sub>esu</sub>lt<sub>s revea</sub>l <sub>a c</sub>l<sub>ear separa</sub>ti<sub>on</sub> b<sub>e</sub>t<sub>ween cr</sub>iti<sub>que genera</sub>ti<sub>on</sub> and context-sensitive judgment: reference-based lexical and <sub>seman</sub>ti<sub>c me</sub>t<sub>r</sub>i<sub>cs on</sub>l<sub>y par</sub>ti<sub>a</sub>ll<sub>y cap</sub>t<sub>ure cr</sub>iti<sub>que qua</sub>lit<sub>y, w</sub>hil<sub>e</sub> <sub>aes</sub>th<sub>e</sub>ti<sub>c spec</sub>i<sub>a</sub>li<sub>s</sub>t<sub>s rema</sub>i<sub>n compe</sub>titi<sub>ve on se</sub>l<sub>ec</sub>t<sub>e</sub>d <sub>cr</sub>iti<sub>que</sub> m<sub>e</sub>tri<sub>cs ye</sub>t <sub>su</sub>b<sub>s</sub>t<sub>a</sub>nti<sub>a</sub>ll<sub>y</sub> l<sub>ag s</sub>tr<sub>o</sub>n<sub>g ge</sub>n<sub>e</sub>r<sub>a</sub>l<sub>-pu</sub>r<sub>pose</sub> MLLM<sub>s</sub> <sub>on</sub> C<sub>ontext</sub>C<sub>anvas.</sub> F<sub>ur</sub>th<sub>er ana</sub>l<sub>yses s</sub>h<sub>ow</sub> th<sub>a</sub>t <sub>aes</sub>th<sub>e</sub>ti<sub>c spe-</sub> <sub>c</sub>i<sub>a</sub>li<sub>za</sub>ti<sub>on</sub> d<sub>oes</sub> <sub>no</sub>t <sub>re</sub>li<sub>a</sub>bl<sub>y</sub> t<sub>rans</sub>f<sub>er</sub> t<sub>o</sub> <sub>con</sub>t<sub>ex</sub>t<sub>ua</sub>l <sub>su</sub>it<sub>a</sub>bilit<sub>y</sub> <sub>an</sub>d th<sub>a</sub>t <sub>mo</sub>d<sub>e</sub>l d<sub>ec</sub>i<sub>s</sub>i<sub>ons</sub> <sub>may</sub> f<sub>a</sub>il t<sub>o</sub> t<sub>rac</sub>k <sub>or</sub> <sub>groun</sub>d th<sub>em-</sub> <sub>se</sub>l<sub>ves</sub> i<sub>n</sub> d<sub>ec</sub>i<sub>s</sub>i<sub>ve con</sub>t<sub>ex</sub>t<sub>ua</sub>l <sub>v</sub>i<sub>sua</sub>l <sub>cues.</sub> Th<sub>ese</sub> fi<sub>n</sub>di<sub>ngs es-</sub> t<sub>a</sub>bli<sub>s</sub>h <sub>cu</sub>lt<sub>ura</sub>ll<sub>y s</sub>it<sub>ua</sub>t<sub>e</sub>d<sub>, ev</sub>id<sub>ence-groun</sub>d<sub>e</sub>d <sub>su</sub>it<sub>a</sub>bilit<sub>y as a</sub> distinct objective for aesthetic modeling.

## I Intr<sub>o</sub>d<sub>uc</sub>ti<sub>o</sub>n

Ima<sub>g</sub>e Aesthetic Assessment (IAA) has <sub>p</sub>ro<sub>g</sub>ressed from scalar <sub>p</sub>rediction and <sub>p</sub>reference modelin<sub>g</sub> (Murra<sub>y</sub>, March-<sub>eso</sub>tti<sub>, an</sub>d P<sub>erronn</sub>i<sub>n</sub> 2012<sub>;</sub> T<sub>a</sub>l<sub>e</sub>bi <sub>an</sub>d Mil<sub>an</sub>f<sub>ar</sub> 2018<sub>;</sub> Yi <sub>e</sub>t <sub>a</sub>l<sub>.</sub> 2023) toward lan<sub>g</sub>ua<sub>g</sub>e-based aesthetic <sub>p</sub>erce<sub>p</sub>tion, critique, dia<sub>g</sub>nosis, and <sub>g</sub>uidance (Huan<sub>g</sub> et al. 2024b,a; Zhou et al. 2024; Zhan<sub>g</sub> et al. 2025; Qi et al. 2025; Cao et al. 2025b). I<sub>n prac</sub>ti<sub>ca</sub>l <sub>use,</sub> h<sub>owever, v</sub>i<sub>sua</sub>l <sub>appea</sub>l <sub>a</sub>l<sub>one</sub> i<sub>s</sub> i<sub>nsu</sub>fi<sub>c</sub>i<sub>en</sub>t<sub>:</sub> <sub>an</sub> i<sub>mage mus</sub>t <sub>a</sub>l<sub>so su</sub>it <sub>a par</sub>ti<sub>cu</sub>l<sub>ar purpose, au</sub>di<sub>ence, an</sub>d <sub>conven</sub>ti<sub>on.</sub> C<sub>u</sub>lt<sub>ura</sub>l k<sub>now</sub>l<sub>e</sub>d<sub>ge</sub> i<sub>s</sub> th<sub>ere</sub>f<sub>ore</sub> <sub>no</sub>t <sub>aux</sub>ili<sub>ary</sub> t<sub>o</sub> aesthetic judgment, since symbols, styles, dress, and genre <sub>conven</sub>ti<sub>ons</sub> <sub>can</sub> di<sub>rec</sub>tl<sub>y</sub> <sub>c</sub>h<sub>ange</sub> <sub>w</sub>h<sub>e</sub>th<sub>er</sub> <sub>an</sub> i<sub>mage</sub> i<sub>s</sub> <sub>an</sub> <sub>ap-</sub> <sub>propr</sub>i<sub>a</sub>t<sub>e v</sub>i<sub>sua</sub>l <sub>c</sub>h<sub>o</sub>i<sub>ce.</sub> R<sub>e</sub>l<sub>a</sub>t<sub>e</sub>d <sub>mu</sub>lti<sub>mo</sub>d<sub>a</sub>l <sub>s</sub>t<sub>u</sub>di<sub>es</sub> lik<sub>ew</sub>i<sub>se</sub> show that models remain fragile when judgments require <sub>cr</sub>it<sub>er</sub>i<sub>on-sens</sub>iti<sub>ve</sub> <sub>reason</sub>i<sub>ng,</sub> <sub>v</sub>i<sub>sua</sub>l <sub>groun</sub>di<sub>ng,</sub> <sub>or</sub> <sub>cu</sub>lt<sub>ura</sub>ll<sub>y</sub>

![](images/f54b6c7435517d2e9e6134c20715d8ac0ef938f8436c0d4055c2ce338816814d.jpg)  
Fi<sub>gu</sub>r<sub>e</sub> 1<sub>:</sub> Fl<sub>u</sub>ent aesthetic criti<sub>qu</sub>e does not <sub>gu</sub>arantee contextually valid judgment. The top example shows an <sub>ev</sub>id<sub>ence-groun</sub>d<sub>e</sub>d <sub>cr</sub>iti<sub>que, w</sub>h<sub>ereas</sub> th<sub>e</sub> b<sub>o</sub>tt<sub>om s</sub>h<sub>ows a</sub> <sub>su</sub>it<sub>a</sub>bilit<sub>y</sub> <sub>error</sub> i<sub>n</sub> <sub>w</sub>hi<sub>c</sub>h <sub>v</sub>i<sub>sua</sub>ll<sub>y</sub> <sub>appea</sub>li<sub>ng</sub> <sub>a</sub>tt<sub>r</sub>ib<sub>u</sub>t<sub>es</sub> <sub>are</sub> <sub>m</sub>i<sub>s</sub>t<sub>a</sub>k<sub>en</sub> f<sub>or</sub> <sub>appropr</sub>i<sub>a</sub>t<sub>eness</sub> i<sub>n</sub> <sub>a</sub> <sub>m</sub>i<sub>sma</sub>t<sub>c</sub>h<sub>e</sub>d <sub>use</sub> <sub>con</sub>t<sub>ex</sub>t<sub>.</sub> Thi<sub>s</sub> <sub>con</sub>t<sub>ras</sub>t <sub>mo</sub>ti<sub>va</sub>t<sub>es</sub> <sub>eva</sub>l<sub>ua</sub>ti<sub>ng</sub> <sub>cr</sub>iti<sub>que</sub> <sub>groun</sub>di<sub>ng</sub> <sub>an</sub>d <sub>con</sub>t<sub>ex</sub>t<sub>ua</sub>l <sub>su</sub>it<sub>a</sub>bilit<sub>y as comp</sub>l<sub>emen</sub>t<sub>ary capa</sub>biliti<sub>es.</sub>

situated evidence (Xion<sub>g</sub> et al. 2026; Li et al. 2026; Wu et al.   
2026b; Satar et al. 2025; Sin<sub>g</sub>h et al. 2026).

Fi<sub>gure</sub> 1 <sub>ma</sub>k<sub>es</sub> thi<sub>s</sub> di<sub>s</sub>ti<sub>nc</sub>ti<sub>on</sub> <sub>concre</sub>t<sub>e:</sub> <sub>a</sub> <sub>mo</sub>d<sub>e</sub>l <sub>may</sub> <sub>pro-</sub> d<sub>uce</sub> <sub>a</sub> <sub>p</sub>l<sub>aus</sub>ibl<sub>e</sub> <sub>aes</sub>th<sub>e</sub>ti<sub>c</sub> <sub>cr</sub>iti<sub>que</sub> <sub>ye</sub>t <sub>approve</sub> <sub>a</sub> <sub>mourn</sub>i<sub>ng</sub> d<sub>ress as</sub> f<sub>es</sub>ti<sub>ve even</sub>i<sub>ng wear</sub> b<sub>y emp</sub>h<sub>as</sub>i<sub>z</sub>i<sub>ng</sub> it<sub>s s</sub>ilh<sub>oue</sub>tt<sub>e</sub> <sub>an</sub>d <sub>ma</sub>t<sub>er</sub>i<sub>a</sub>l <sub>w</sub>hil<sub>e</sub> i<sub>gnor</sub>i<sub>ng</sub> it<sub>s soc</sub>i<sub>a</sub>l f<sub>unc</sub>ti<sub>on.</sub> Fi<sub>gure</sub> 2 f<sub>ur-</sub> th<sub>er revea</sub>l<sub>s</sub> t<sub>wo</sub> f<sub>a</sub>il<sub>ure</sub> l<sub>eve</sub>l<sub>s: cr</sub>iti<sub>ques may re</sub>l<sub>y on</sub> f<sub>a</sub>b<sub>r</sub>i<sub>-</sub> <sub>ca</sub>t<sub>e</sub>d <sub>ev</sub>id<sub>ence or</sub> i<sub>nappropr</sub>i<sub>a</sub>t<sub>e</sub> d<sub>oma</sub>i<sub>n pr</sub>i<sub>ors, w</sub>hil<sub>e su</sub>it<sub>-</sub> ability judgments may privilege surface appeal or stylistic plausibility over cultural fit. We call the latter tendency aesthetic context bias: following generic aesthetic priors while <sub>neg</sub>l<sub>ec</sub>ti<sub>ng con</sub>t<sub>ex</sub>t<sub>ua</sub>l <sub>ev</sub>id<sub>ence</sub> th<sub>a</sub>t <sub>s</sub>h<sub>ou</sub>ld <sub>a</sub>lt<sub>er</sub> th<sub>e</sub> d<sub>ec</sub>i<sub>s</sub>i<sub>on.</sub> Thi<sub>s</sub> t<sub>as</sub>k i<sub>s no</sub>t <sub>gener</sub>i<sub>c cu</sub>lt<sub>ura</sub>l <sub>ques</sub>ti<sub>on answer</sub>i<sub>ng; con</sub>t<sub>ex-</sub> t<sub>ua</sub>l k<sub>now</sub>l<sub>e</sub>d<sub>ge ma</sub>tt<sub>ers</sub> b<sub>ecause</sub> it d<sub>e</sub>t<sub>erm</sub>i<sub>nes w</sub>h<sub>e</sub>th<sub>er</sub> th<sub>e</sub> i<sub>mage wor</sub>k<sub>s as a v</sub>i<sub>sua</sub>l <sub>c</sub>h<sub>o</sub>i<sub>ce.</sub>

![](images/f6d40aba9cb3b8f0a579935507cd767f1f250d0af0ea2fe35f2ada3e0f09a34a.jpg)  
Fi<sub>gure</sub> 2<sub>:</sub> F<sub>ou</sub>r r<sub>ecu</sub>rrin<sub>g</sub> f<sub>a</sub>il<sub>u</sub>r<sub>e</sub> m<sub>o</sub>d<sub>es</sub> in <sub>p</sub>r<sub>ac</sub>ti<sub>ca</sub>l <sub>aes</sub>th<sub>e</sub>ti<sub>c</sub> <sub>eva</sub>l<sub>ua</sub>ti<sub>o</sub>n<sub>.</sub> M<sub>o</sub>d<sub>e</sub>l<sub>s</sub> <sub>may</sub> <sub>genera</sub>t<sub>e</sub> <sub>cr</sub>iti<sub>ques</sub> th<sub>a</sub>t <sub>are</sub> <sub>v</sub>i<sub>sua</sub>ll<sub>y</sub> ungrounded or rely on inappropriate domain priors, and may judge real-world suitability from surface aesthetics or stylistic <sub>p</sub>l<sub>aus</sub>ibilit<sub>y ra</sub>th<sub>er</sub> th<sub>an commun</sub>i<sub>ca</sub>ti<sub>ve an</sub>d <sub>con</sub>t<sub>ex</sub>t<sub>ua</sub>l fit<sub>.</sub>

T<sub>o s</sub>t<sub>u</sub>d<sub>y</sub> b<sub>o</sub>th <sub>capa</sub>biliti<sub>es, we</sub> i<sub>n</sub>t<sub>ro</sub>d<sub>uce</sub> A<sub>es</sub>C<sub>anvas, a</sub> <sub>un</sub>ifi<sub>e</sub>d <sub>su</sub>it<sub>e w</sub>ith t<sub>wo comp</sub>l<sub>emen</sub>t<sub>ary componen</sub>t<sub>s.</sub> C<sub>ri-</sub> <sub>tique</sub>C<sub>anvas prov</sub>id<sub>es</sub> l<sub>arge-sca</sub>l<sub>e, mu</sub>lti<sub>-</sub>d<sub>oma</sub>i<sub>n superv</sub>i<sub>-</sub> <sub>s</sub>i<sub>on</sub> f<sub>or</sub> l<sub>ong-</sub>f<sub>orm</sub> <sub>cr</sub>iti<sub>que</sub> <sub>across</sub> <sub>p</sub>h<sub>o</sub>t<sub>ograp</sub>h<sub>y,</sub> <sub>pa</sub>i<sub>n</sub>ti<sub>ng,</sub> <sub>an</sub>d <sub>v</sub>i<sub>r</sub>t<sub>ua</sub>l i<sub>magery, w</sub>hil<sub>e</sub> C<sub>ontext</sub>C<sub>anvas con</sub>t<sub>a</sub>i<sub>ns exper</sub>t<sub>-</sub> <sub>rev</sub>i<sub>ewe</sub>d <sub>use scenar</sub>i<sub>os</sub> th<sub>a</sub>t di<sub>s</sub>ti<sub>ngu</sub>i<sub>s</sub>h <sub>v</sub>i<sub>sua</sub>l <sub>appea</sub>l f<sub>rom</sub> <sub>con</sub>t<sub>ex</sub>t<sub>ua</sub>l <sub>su</sub>it<sub>a</sub>bilit<sub>y.</sub> A<sub>cross</sub> <sub>c</sub>l<sub>ose</sub>d<sub>-source,</sub> <sub>open-we</sub>i<sub>g</sub>ht<sub>,</sub> <sub>an</sub>d <sub>aes</sub>th<sub>e</sub>ti<sub>c-spec</sub>ifi<sub>c</sub> MLLM<sub>s, we o</sub>b<sub>serve a c</sub>l<sub>ear capa</sub>bilit<sub>y</sub> <sub>gap: aes</sub>th<sub>e</sub>ti<sub>c spec</sub>i<sub>a</sub>li<sub>s</sub>t<sub>s rema</sub>i<sub>n compe</sub>titi<sub>ve on se</sub>l<sub>ec</sub>t<sub>e</sub>d <sub>cr</sub>i<sub>-</sub> ti<sub>que</sub> m<sub>e</sub>tri<sub>cs, ye</sub>t <sub>a</sub>ll <sub>sco</sub>r<sub>e</sub> b<sub>e</sub>l<sub>ow</sub> 30% <sub>o</sub>n C<sub>ontext</sub>C<sub>anvas,</sub> <sub>compare</sub>d <sub>w</sub>ith 91<sub>.</sub>69% f<sub>or</sub> th<sub>e s</sub>t<sub>ronges</sub>t <sub>eva</sub>l<sub>ua</sub>t<sub>e</sub>d <sub>genera</sub>l<sub>-</sub> <sub>purpose mo</sub>d<sub>e</sub>l<sub>.</sub> F<sub>ur</sub>th<sub>er ana</sub>l<sub>yses s</sub>h<sub>ow</sub> th<sub>a</sub>t <sub>re</sub>f<sub>erence-</sub>b<sub>ase</sub>d <sub>me</sub>t<sub>r</sub>i<sub>cs on</sub>l<sub>y par</sub>ti<sub>a</sub>ll<sub>y cap</sub>t<sub>ure cr</sub>iti<sub>que qua</sub>lit<sub>y, aes</sub>th<sub>e</sub>ti<sub>c spe-</sub> cialization does not reliably transfer to contextual judgment, <sub>an</sub>d <sub>correc</sub>t d<sub>ec</sub>i<sub>s</sub>i<sub>ons</sub> <sub>may</sub> <sub>s</sub>till l<sub>ac</sub>k <sub>groun</sub>di<sub>ng</sub> i<sub>n</sub> d<sub>ec</sub>i<sub>s</sub>i<sub>ve</sub> <sub>v</sub>i<sub>sua</sub>l <sub>ev</sub>id<sub>ence.</sub>

Contributions. (1) We introduce CritiqueCanvas, com-<sub>pr</sub>i<sub>s</sub>i<sub>ng</sub> 519<sub>,</sub>136 <sub>aes</sub>th<sub>e</sub>ti<sub>c</sub> i<sub>ns</sub>t<sub>ruc</sub>ti<sub>on–response</sub> <sub>pa</sub>i<sub>rs</sub> b<sub>u</sub>ilt f<sub>rom</sub> 54<sub>,</sub>300 <sub>mu</sub>lti<sub>-</sub>d<sub>oma</sub>i<sub>n</sub> i<sub>mages</sub> f<sub>or</sub> i<sub>ns</sub>t<sub>ruc</sub>ti<sub>on-</sub>t<sub>un</sub>i<sub>ng</sub> MLLMs on fine-<sub>g</sub>rained ima<sub>g</sub>e aesthetic assessment; (2) <sub>we</sub> <sub>co</sub>n<sub>s</sub>tr<sub>uc</sub>t C<sub>ontext</sub>C<sub>anvas,</sub> <sub>a</sub> 301<sub>-case</sub> <sub>expe</sub>rt<sub>-cu</sub>r<sub>a</sub>t<sub>e</sub>d benchmark for contextual aesthetic suitability judgment; (3) we evaluate closed-source, o<sub>p</sub>en-wei<sub>g</sub>ht, and aesthetic-<sub>spec</sub>ifi<sub>c</sub> MLLM<sub>s,</sub> r<sub>evea</sub>lin<sub>g</sub> <sub>a</sub> <sub>c</sub>l<sub>ea</sub>r <sub>gap</sub> b<sub>e</sub>t<sub>wee</sub>n <sub>p</sub>r<sub>o</sub>d<sub>uc</sub>in<sub>g</sub> fluent aesthetic critiques and correctly judging suitability <sub>un</sub>d<sub>er cu</sub>lt<sub>ura</sub>l<sub>,</sub> f<sub>unc</sub>ti<sub>ona</sub>l<sub>, an</sub>d d<sub>oma</sub>i<sub>n-spec</sub>ifi<sub>c cons</sub>t<sub>ra</sub>i<sub>n</sub>t<sub>s;</sub> and (4) we release the data, <sub>p</sub>rom<sub>p</sub>ts, rationales, <sub>p</sub>rovenance <sub>recor</sub>d<sub>s, an</sub>d <sub>eva</sub>l<sub>ua</sub>ti<sub>on co</sub>d<sub>e.</sub>

## II R<sub>e</sub>l<sub>a</sub>t<sub>e</sub>d W<sub>or</sub>k

## A Im<sub>age</sub> A<sub>es</sub>th<sub>e</sub>ti<sub>c</sub> A<sub>ssess</sub>m<sub>e</sub>nt

Ima<sub>g</sub>e Aesthetic Assessment (IAA) <sub>p</sub>redicts human jud<sub>g</sub>- <sub>men</sub>t<sub>s o</sub>f <sub>v</sub>i<sub>sua</sub>l <sub>aes</sub>th<sub>e</sub>ti<sub>c qua</sub>lit<sub>y.</sub> E<sub>ar</sub>l<sub>y wor</sub>k f<sub>ormu</sub>l<sub>a</sub>t<sub>e</sub>d IAA <sub>as c</sub>l<sub>ass</sub>ifi<sub>ca</sub>ti<sub>on, ran</sub>ki<sub>ng, or score-</sub>di<sub>s</sub>t<sub>r</sub>ib<sub>u</sub>ti<sub>on pre</sub>di<sub>c</sub>ti<sub>on</sub> f<sub>rom crow</sub>d <sub>pre</sub>f<sub>erences, w</sub>hil<sub>e</sub> i<sub>ncorpora</sub>ti<sub>ng p</sub>h<sub>o</sub>t<sub>ograp</sub>hi<sub>c</sub> <sub>a</sub>tt<sub>r</sub>ib<sub>u</sub>t<sub>es, compos</sub>iti<sub>on-preserv</sub>i<sub>ng represen</sub>t<sub>a</sub>ti<sub>ons, an</sub>d i<sub>n-</sub> dividual taste variation (Murra<sub>y</sub>, Marchesotti, and Perronnin 2012<sub>;</sub> K<sub>ong</sub> <sub>e</sub>t <sub>a</sub>l<sub>.</sub> 2016<sub>;</sub> M<sub>a</sub>i<sub>,</sub> Ji<sub>n,</sub> <sub>an</sub>d Li<sub>u</sub> 2016<sub>;</sub> Y<sub>ang</sub> <sub>e</sub>t <sub>a</sub>l<sub>.</sub> 2022). Recent studies extend IAA to artistic ima<sub>g</sub>es, fine-<sub>gra</sub>i<sub>ne</sub>d <sub>compar</sub>i<sub>son</sub> <sub>w</sub>ithi<sub>n</sub> i<sub>mage</sub> <sub>ser</sub>i<sub>es,</sub> <sub>an</sub>d <sub>open-wor</sub>ld <sub>cr</sub>i<sub>-</sub> teria that var<sub>y</sub> across themes (Yi et al. 2023; Yan<sub>g</sub> et al. 2026; Liao, Ma, and Zhan<sub>g</sub> 2026). Nevertheless, these methods <sub>p</sub>ri-<sub>mar</sub>il<sub>y assess aes</sub>th<sub>e</sub>ti<sub>c s</sub>t<sub>reng</sub>th <sub>un</sub>d<sub>er</sub> i<sub>n</sub>t<sub>r</sub>i<sub>ns</sub>i<sub>c or</sub> t<sub>as</sub>k<sub>-</sub>fi<sub>xe</sub>d <sub>cr</sub>it<sub>er</sub>i<sub>a,</sub> <sub>ra</sub>th<sub>er</sub> th<sub>an</sub> <sub>w</sub>h<sub>e</sub>th<sub>er</sub> <sub>an</sub> i<sub>mage</sub> i<sub>s</sub> <sub>appropr</sub>i<sub>a</sub>t<sub>e</sub> f<sub>or</sub> <sub>a</sub> <sub>par</sub>ti<sub>cu</sub>l<sub>ar purpose, au</sub>di<sub>ence, cu</sub>lt<sub>ura</sub>l <sub>se</sub>tti<sub>ng, or</sub> d<sub>oma</sub>i<sub>n con-</sub> <sub>ven</sub>ti<sub>on.</sub> O<sub>ur wor</sub>k <sub>a</sub>dd<sub>resses</sub> thi<sub>s comp</sub>l<sub>emen</sub>t<sub>ary pro</sub>bl<sub>em o</sub>f <sub>con</sub>t<sub>ex</sub>t<sub>ua</sub>l <sub>aes</sub>th<sub>e</sub>ti<sub>c su</sub>it<sub>a</sub>bilit<sub>y.</sub>

## B A<sub>es</sub>th<sub>e</sub>ti<sub>c</sub> M<sub>u</sub>lti<sub>mo</sub>d<sub>a</sub>l M<sub>o</sub>d<sub>e</sub>l<sub>s</sub>

A<sub>es</sub>th<sub>e</sub>ti<sub>c mu</sub>lti<sub>mo</sub>d<sub>a</sub>l <sub>mo</sub>d<sub>e</sub>l<sub>s ex</sub>t<sub>en</sub>d <sub>conven</sub>ti<sub>ona</sub>l IAA f<sub>rom</sub> <sub>sca</sub>l<sub>ar pre</sub>di<sub>c</sub>ti<sub>on</sub> t<sub>o</sub> l<sub>anguage-</sub>b<sub>ase</sub>d <sub>recogn</sub>iti<sub>on,</sub> d<sub>escr</sub>i<sub>p</sub>ti<sub>on,</sub> i<sub>n</sub>t<sub>erpre</sub>t<sub>a</sub>ti<sub>on, scor</sub>i<sub>ng, an</sub>d <sub>cr</sub>iti<sub>que.</sub> E<sub>ar</sub>l<sub>y s</sub>t<sub>u</sub>di<sub>es</sub> b<sub>enc</sub>h<sub>-</sub> m<sub>a</sub>rk<sub>e</sub>d th<sub>e aes</sub>th<sub>e</sub>ti<sub>c pe</sub>r<sub>cep</sub>ti<sub>o</sub>n <sub>o</sub>f <sub>ge</sub>n<sub>e</sub>r<sub>a</sub>l<sub>-pu</sub>r<sub>pose</sub> MLLM<sub>s</sub> and ex<sub>p</sub>lored aesthetic instruction tunin<sub>g</sub> (Huan<sub>g</sub> et al. 2024b,a; Zhou et al. 2024). Subsequent work connected t<sub>ex</sub>t<sub>ua</sub>l <sub>superv</sub>i<sub>s</sub>i<sub>on</sub> <sub>w</sub>ith <sub>v</sub>i<sub>sua</sub>l <sub>scor</sub>i<sub>ng</sub> th<sub>roug</sub>h l<sub>anguage-</sub> defined rating levels or joint scoring-and-interpretation objectives (Wu et al. 2024; Zhan<sub>g</sub> et al. 2025), while more <sub>recen</sub>t <sub>mo</sub>d<sub>e</sub>l<sub>s suppor</sub>t <sub>pro</sub>f<sub>ess</sub>i<sub>ona</sub>l <sub>cr</sub>iti<sub>que, mu</sub>lti<sub>-a</sub>tt<sub>r</sub>ib<sub>u</sub>t<sub>e</sub> anal<sub>y</sub>sis, and unified perceptual assessment (Qi et al. 2025; Cao et al. 2025b,a). Existin<sub>g</sub> evaluation, however, remains <sub>cen</sub>t<sub>ere</sub>d <sub>on aes</sub>th<sub>e</sub>ti<sub>c percep</sub>ti<sub>on, scor</sub>i<sub>ng, re</sub>f<sub>erence-a</sub>li<sub>gne</sub>d <sub>cr</sub>iti<sub>que, or</sub> d<sub>oma</sub>i<sub>n-spec</sub>ifi<sub>c ana</sub>l<sub>ys</sub>i<sub>s.</sub> A<sub>es</sub>C<sub>anvas</sub> i<sub>ns</sub>t<sub>ea</sub>d <sub>ex-</sub> <sub>am</sub>i<sub>nes w</sub>h<sub>e</sub>th<sub>er aes</sub>th<sub>e</sub>ti<sub>c ar</sub>ti<sub>cu</sub>l<sub>a</sub>ti<sub>on</sub> t<sub>rans</sub>f<sub>ers</sub> t<sub>o con</sub>t<sub>ex</sub>t<sub>ua</sub>l d<sub>ec</sub>i<sub>s</sub>i<sub>ons</sub> i<sub>n w</sub>hi<sub>c</sub>h <sub>an a</sub>tt<sub>rac</sub>ti<sub>ve or s</sub>t<sub>y</sub>li<sub>s</sub>ti<sub>ca</sub>ll<sub>y p</sub>l<sub>aus</sub>ibl<sub>e</sub> i<sub>m-</sub> <sub>age may s</sub>till <sub>con</sub>fli<sub>c</sub>t <sub>w</sub>ith it<sub>s</sub> i<sub>n</sub>t<sub>en</sub>d<sub>e</sub>d <sub>use.</sub> B<sub>eyon</sub>d <sub>con-</sub> <sub>ven</sub>ti<sub>ona</sub>l <sub>v</sub>i<sub>sua</sub>l i<sub>npu</sub>t<sub>s, recen</sub>t <sub>mu</sub>lti<sub>mo</sub>d<sub>a</sub>l f<sub>oun</sub>d<sub>a</sub>ti<sub>on mo</sub>d<sub>-</sub> <sub>e</sub>l<sub>s</sub> h<sub>ave</sub> <sub>a</sub>l<sub>so</sub> <sub>exp</sub>l<sub>ore</sub>d <sub>seman</sub>ti<sub>c</sub> <sub>un</sub>d<sub>ers</sub>t<sub>an</sub>di<sub>ng</sub> <sub>an</sub>d l<sub>anguage</sub> <sub>genera</sub>ti<sub>on over spec</sub>i<sub>a</sub>li<sub>ze</sub>d <sub>or corrup</sub>t<sub>e</sub>d <sub>v</sub>i<sub>sua</sub>l <sub>represen</sub>t<sub>a-</sub> tions (Wu et al. 2026a; Lian et al. 2026).

![](images/715edb21500898030ffb7071c9a218120cd3bbff4ce48c0547eff08fd319b495.jpg)  
Fi ure 3: Construction of AesCanvas. (a) A shared multi-domain ima e ool is collected and filtered. (b) CritiqueCanvas is <sub>g</sub>enerated throu<sub>g</sub>h dimension-aware <sub>p</sub>rom<sub>p</sub>t <sub>p</sub>lannin<sub>g</sub>, MLLM-assisted annotation, and qualit<sub>y</sub> filterin<sub>g</sub>. (c) ContextCanvas i<sub>s cons</sub>t<sub>ruc</sub>t<sub>e</sub>d f<sub>rom rea</sub>li<sub>s</sub>ti<sub>c aes</sub>th<sub>e</sub>ti<sub>c-use scenar</sub>i<sub>os</sub> f<sub>o</sub>ll<sub>owe</sub>d b<sub>y exper</sub>t <sub>rev</sub>i<sub>ew.</sub>

## C C<sub>o</sub>nt<sub>ex</sub>t<sub>ua</sub>l M<sub>u</sub>ltim<sub>o</sub>d<sub>a</sub>l R<sub>easo</sub>nin<sub>g</sub>

C<sub>on</sub>t<sub>ex</sub>t<sub>ua</sub>l <sub>mu</sub>lti<sub>mo</sub>d<sub>a</sub>l <sub>reason</sub>i<sub>ng requ</sub>i<sub>res mo</sub>d<sub>e</sub>l<sub>s</sub> t<sub>o com-</sub> bi<sub>ne v</sub>i<sub>sua</sub>l <sub>ev</sub>id<sub>ence w</sub>ith t<sub>as</sub>k <sub>con</sub>diti<sub>ons, ex</sub>t<sub>erna</sub>l k<sub>now</sub>l<sub>-</sub> <sub>e</sub>d<sub>ge, cu</sub>lt<sub>ura</sub>l <sub>conven</sub>ti<sub>ons, an</sub>d <sub>cr</sub>it<sub>er</sub>i<sub>on-spec</sub>ifi<sub>c cons</sub>t<sub>ra</sub>i<sub>n</sub>t<sub>s.</sub> R<sub>ecen</sub>t b<sub>enc</sub>h<sub>mar</sub>k<sub>s s</sub>t<sub>u</sub>d<sub>y rea</sub>l<sub>-wor</sub>ld <sub>v</sub>i<sub>sua</sub>l <sub>reason</sub>i<sub>ng,</sub> context-dependent functional inference, and flexible judgment under var<sub>y</sub>in<sub>g</sub> criteria (Yin et al. 2026; Zhan<sub>g</sub> et al. 2026; Xion<sub>g</sub> et al. 2026; Gao et al. 2024). Groundin<sub>g</sub>-focused <sub>wor</sub>k f<sub>ur</sub>th<sub>er eva</sub>l<sub>ua</sub>t<sub>es w</sub>h<sub>e</sub>th<sub>er</sub> i<sub>n</sub>t<sub>erme</sub>di<sub>a</sub>t<sub>e reason</sub>i<sub>ng an</sub>d final answers are su<sub>pp</sub>orted b<sub>y</sub> localized ima<sub>g</sub>e evidence (Li et al. 2026; Wu et al. 2026b; Cai et al. 2024), while other <sub>s</sub>t<sub>u</sub>di<sub>es expose</sub> f<sub>a</sub>il<sub>ures un</sub>d<sub>er v</sub>i<sub>sua</sub>l ill<sub>us</sub>i<sub>ons, coun</sub>t<sub>er</sub>f<sub>ac</sub>t<sub>ua</sub>l <sub>or a</sub>d<sub>versar</sub>i<sub>a</sub>l <sub>ev</sub>id<sub>ence, an</sub>d <sub>con</sub>fli<sub>c</sub>t<sub>s</sub> b<sub>e</sub>t<sub>ween</sub> k<sub>now</sub>l<sub>e</sub>d<sub>ge</sub> and ima<sub>g</sub>es (Hou et al. 2026; Zhou et al. 2026; Moratelli et al. 2026). Culturall<sub>y</sub> situated benchmarks likewise reveal f<sub>rag</sub>ilit<sub>y on reg</sub>i<sub>on-spec</sub>ifi<sub>c ar</sub>tif<sub>ac</sub>t<sub>s an</sub>d <sub>mu</sub>ltili<sub>ngua</sub>l <sub>cu</sub>lt<sub>ura</sub>l evidence (Satar et al. 2025; Sin<sub>g</sub>h et al. 2026). However, th<sub>ese</sub> <sub>wor</sub>k<sub>s</sub> <sub>pr</sub>i<sub>mar</sub>il<sub>y</sub> <sub>assess</sub> f<sub>ac</sub>t<sub>ua</sub>l <sub>correc</sub>t<sub>ness,</sub> <sub>groun</sub>d<sub>-</sub> i<sub>ng,</sub> f<sub>unc</sub>ti<sub>ona</sub>l i<sub>n</sub>f<sub>erence, or a</sub>dh<sub>erence</sub> t<sub>o spec</sub>ifi<sub>e</sub>d <sub>cr</sub>it<sub>er</sub>i<sub>a,</sub> <sub>ra</sub>th<sub>er</sub> th<sub>an w</sub>h<sub>e</sub>th<sub>er con</sub>t<sub>ex</sub>t<sub>ua</sub>l <sub>ev</sub>id<sub>ence s</sub>h<sub>ou</sub>ld <sub>ou</sub>t<sub>we</sub>i<sub>g</sub>h <sub>sur-</sub> f<sub>ace aes</sub>th<sub>e</sub>ti<sub>c appea</sub>l i<sub>n a concre</sub>t<sub>e v</sub>i<sub>sua</sub>l<sub>-use</sub> d<sub>ec</sub>i<sub>s</sub>i<sub>on.</sub> C<sub>on-</sub> <sub>text</sub>C<sub>anvas</sub> t<sub>arge</sub>t<sub>s</sub> thi<sub>s</sub> i<sub>n</sub>t<sub>ersec</sub>ti<sub>on</sub> th<sub>roug</sub>h <sub>use-or</sub>i<sub>en</sub>t<sub>e</sub>d aesthetic suitability judgments grounded in both visual and <sub>con</sub>t<sub>ex</sub>t<sub>ua</sub>l <sub>ev</sub>id<sub>ence.</sub>

## III D<sub>a</sub>t<sub>ase</sub>t

## A D<sub>a</sub>t<sub>ase</sub>t O<sub>ve</sub>r<sub>v</sub>i<sub>ew</sub>

W<sub>e</sub> i<sub>n</sub>t<sub>ro</sub>d<sub>uce</sub> A<sub>es</sub>C<sub>anvas, a un</sub>ifi<sub>e</sub>d <sub>su</sub>it<sub>e</sub> b<sub>u</sub>ilt f<sub>rom</sub> 54<sub>,</sub>300 i<sub>mages</sub> t<sub>o s</sub>t<sub>u</sub>d<sub>y</sub> t<sub>wo comp</sub>l<sub>emen</sub>t<sub>ary aspec</sub>t<sub>s o</sub>f <sub>aes</sub>th<sub>e</sub>ti<sub>c</sub> intelligence: articulating visual qualities and judging their <sub>su</sub>it<sub>a</sub>bilit<sub>y</sub> i<sub>n</sub> <sub>con</sub>t<sub>ex</sub>t<sub>.</sub> C<sub>ritique</sub>C<sub>anvas</sub> <sub>con</sub>t<sub>a</sub>i<sub>ns</sub> 519<sub>,</sub>136 <sup>instr</sup>u<sup>ction–res</sup>p<sup>onse</sup> p<sup>airs</sup> <sup>across</sup> p<sup>hoto</sup>g<sup>ra</sup>p<sup>h</sup>y, p<sup>aintin</sup>g, <sub>an</sub>d <sub>v</sub>i<sub>r</sub>t<sub>ua</sub>l i<sub>magery</sub> f<sub>or</sub> l<sub>ong-</sub>f<sub>orm, mu</sub>lti<sub>-</sub>di<sub>mens</sub>i<sub>ona</sub>l <sub>cr</sub>i<sub>-</sub> ti<sub>que</sub> <sub>ge</sub>n<sub>e</sub>r<sub>a</sub>ti<sub>o</sub>n<sub>.</sub> C<sub>ontext</sub>C<sub>anvas</sub> <sub>co</sub>m<sub>p</sub>ri<sub>ses</sub> 301 <sub>expe</sub>rt<sub>-</sub> d<sub>es</sub>i<sub>gne</sub>d <sub>an</sub>d <sub>rev</sub>i<sub>ewe</sub>d <sub>cases</sub> th<sub>a</sub>t <sub>eva</sub>l<sub>ua</sub>t<sub>e con</sub>t<sub>ex</sub>t<sub>ua</sub>l <sub>aes-</sub> th<sub>e</sub>ti<sub>c</sub> <sub>su</sub>it<sub>a</sub>bilit<sub>y—w</sub>h<sub>e</sub>th<sub>er</sub> <sub>an</sub> i<sub>mage</sub> i<sub>s</sub> <sub>an</sub> <sub>appropr</sub>i<sub>a</sub>t<sub>e</sub> <sub>v</sub>i<sub>sua</sub>l <sub>c</sub>h<sub>o</sub>i<sub>ce</sub> f<sub>or</sub> <sub>a</sub> <sub>spec</sub>ifi<sub>e</sub>d <sub>purpose,</sub> <sub>au</sub>di<sub>ence,</sub> <sub>cu</sub>lt<sub>ura</sub>l <sub>se</sub>tti<sub>ng,</sub> <sub>or</sub> d<sub>oma</sub>i<sub>n</sub> <sub>conven</sub>ti<sub>on,</sub> b<sub>eyon</sub>d it<sub>s</sub> i<sub>n</sub>t<sub>r</sub>i<sub>ns</sub>i<sub>c</sub> <sub>v</sub>i<sub>sua</sub>l <sub>appea</sub>l<sub>.</sub> T<sub>o-</sub> <sub>ge</sub>th<sub>er,</sub> th<sub>e</sub> t<sub>wo componen</sub>t<sub>s</sub> t<sub>es</sub>t <sub>w</sub>h<sub>e</sub>th<sub>er mo</sub>d<sub>e</sub>l<sub>s</sub> th<sub>a</sub>t <sub>can</sub> id<sub>en</sub>tif<sub>y an</sub>d <sub>ar</sub>ti<sub>cu</sub>l<sub>a</sub>t<sub>e aes</sub>th<sub>e</sub>ti<sub>c ev</sub>id<sub>ence can a</sub>l<sub>so</sub> t<sub>rans</sub>l<sub>a</sub>t<sub>e</sub> it i<sub>n</sub>t<sub>o con</sub>t<sub>ex</sub>t<sub>ua</sub>ll<sub>y va</sub>lid d<sub>ec</sub>i<sub>s</sub>i<sub>ons.</sub>

## B D<sub>a</sub>t<sub>ase</sub>t C<sub>u</sub>r<sub>a</sub>ti<sub>o</sub>n

A<sub>s</sub> ill<sub>us</sub>t<sub>ra</sub>t<sub>e</sub>d i<sub>n</sub> Fi<sub>g.</sub> 3<sub>, our p</sub>i<sub>pe</sub>li<sub>ne</sub> b<sub>ranc</sub>h<sub>es</sub> f<sub>rom a s</sub>h<sub>are</sub>d <sub>mu</sub>lti<sub>-</sub>d<sub>oma</sub>i<sub>n</sub> i<sub>mage poo</sub>l i<sub>n</sub>t<sub>o</sub> C<sub>ritique</sub>C<sub>anvas</sub> f<sub>or s</sub>t<sub>ruc-</sub> t<sub>ure</sub>d <sub>aes</sub>th<sub>e</sub>ti<sub>c cr</sub>iti<sub>que an</sub>d C<sub>ontext</sub>C<sub>anvas</sub> f<sub>or c</sub>l<sub>ose</sub>d<sub>-</sub>f<sub>orm</sub> contextual suitability judgment.

Im<sub>age</sub> C<sub>o</sub>ll<sub>ec</sub>ti<sub>o</sub>n <sub>a</sub>nd Filt<sub>e</sub>rin<sub>g.</sub> W<sub>e</sub> <sub>co</sub>ll<sub>ec</sub>t <sub>can</sub>did<sub>a</sub>t<sub>e</sub> i<sub>m-</sub> ages from three visual domains: photography (e.g., natural scenes, architecture, <sub>p</sub>ortraiture, and commercial ima<sub>g</sub>er<sub>y</sub>), painting (primarily Western classical and related works), and virtual imagery (e.g., game scenes, animation-style images, and AI-<sub>g</sub>enerated content). Ima<sub>g</sub>es are drawn from <sub>p</sub>ublicl<sub>y</sub> <sub>access</sub>ibl<sub>e we</sub>b <sub>sources an</sub>d <sub>on</sub>li<sub>ne museum an</sub>d <sub>cu</sub>lt<sub>ura</sub>l<sub>-</sub> h<sub>er</sub>it<sub>age co</sub>ll<sub>ec</sub>ti<sub>ons, w</sub>ith <sub>ava</sub>il<sub>a</sub>bl<sub>e source</sub> URL<sub>s an</sub>d <sub>prove-</sub> <sub>nance me</sub>t<sub>a</sub>d<sub>a</sub>t<sub>a re</sub>t<sub>a</sub>i<sub>ne</sub>d<sub>.</sub> Aft<sub>er remov</sub>i<sub>ng</sub> l<sub>ow-qua</sub>lit<sub>y, seman-</sub> ti<sub>ca</sub>ll<sub>y un</sub>i<sub>n</sub>f<sub>orma</sub>ti<sub>ve, non-comp</sub>li<sub>an</sub>t<sub>, an</sub>d d<sub>up</sub>li<sub>ca</sub>t<sub>e samp</sub>l<sub>es,</sub> th<sub>e poo</sub>l <sub>con</sub>t<sub>a</sub>i<sub>ns</sub> 21<sub>,</sub>294 <sub>p</sub>h<sub>o</sub>t<sub>ograp</sub>h<sub>s,</sub> 17<sub>,</sub>939 <sub>pa</sub>i<sub>n</sub>ti<sub>ngs, an</sub>d 15<sub>,</sub>067 <sub>v</sub>i<sub>r</sub>t<sub>ua</sub>l i<sub>mages,</sub> f<sub>or a</sub> t<sub>o</sub>t<sub>a</sub>l <sub>o</sub>f 54<sub>,</sub>300<sub>.</sub>

Criti<sub>qu</sub>eCa<sub>nv</sub>a<sub>s</sub> Ann<sub>o</sub>t<sub>a</sub>ti<sub>o</sub>n <sub>a</sub>nd C<sub>o</sub>n<sub>s</sub>tr<sub>uc</sub>ti<sub>o</sub>n<sub>.</sub> A<sub>s</sub> shown in Fi<sub>g</sub>. 3(b), we use a <sub>p</sub>rom<sub>p</sub>t-based <sub>p</sub>i<sub>p</sub>eline that <sub>com</sub>bi<sub>nes s</sub>h<sub>are</sub>d <sub>aes</sub>th<sub>e</sub>ti<sub>c</sub> di<sub>mens</sub>i<sub>ons w</sub>ith d<sub>oma</sub>i<sub>n-spec</sub>ifi<sub>c</sub> criteria. The shared dimensions include Content and Narrative, Composition, Color, Lighting, Lines and Brushstrokes, Style, Emotion, Technique, Symbolism, and Visual Appeal. A<sub>nno</sub>t<sub>a</sub>ti<sub>ng mo</sub>d<sub>e</sub>l<sub>s se</sub>l<sub>ec</sub>t <sub>on</sub>l<sub>y</sub> th<sub>e</sub> di<sub>mens</sub>i<sub>ons mos</sub>t <sub>re</sub>l<sub>e-</sub> <sub>van</sub>t t<sub>o eac</sub>h i<sub>mage ra</sub>th<sub>er</sub> th<sub>an a</sub>dd<sub>ress</sub>i<sub>ng a</sub>ll di<sub>mens</sub>i<sub>ons</sub> <sub>un</sub>if<sub>orm</sub>l<sub>y.</sub>

P<sub>romp</sub>t<sub>s are</sub> f<sub>ur</sub>th<sub>er con</sub>diti<sub>one</sub>d <sub>on</sub> th<sub>e v</sub>i<sub>sua</sub>l d<sub>oma</sub>i<sub>n.</sub> Ph<sub>o</sub>t<sub>og</sub>r<sub>ap</sub>h<sub>y</sub> <sub>e</sub>m<sub>p</sub>h<sub>as</sub>i<sub>zes</sub> <sub>exposu</sub>r<sub>e,</sub> <sub>pe</sub>r<sub>spec</sub>ti<sub>ve,</sub> d<sub>ep</sub>th <sub>o</sub>f fi<sub>e</sub>ld<sub>,</sub> f<sub>ocus,</sub> <sub>mo</sub>ti<sub>on,</sub> <sub>an</sub>d <sub>compos</sub>iti<sub>ona</sub>l <sub>con</sub>t<sub>ro</sub>l<sub>;</sub> <sub>pa</sub>i<sub>n</sub>ti<sub>ng</sub> <sub>em-</sub> <sub>p</sub>h<sub>as</sub>i<sub>zes</sub> li<sub>ne,</sub> b<sub>rus</sub>h<sub>wor</sub>k<sub>,</sub> <sub>genre,</sub> <sub>s</sub>t<sub>y</sub>l<sub>e,</sub> <sub>sym</sub>b<sub>o</sub>li<sub>sm,</sub> <sub>an</sub>d <sub>emo-</sub> ti<sub>ona</sub>l <sub>express</sub>i<sub>on; an</sub>d <sub>v</sub>i<sub>r</sub>t<sub>ua</sub>l i<sub>magery emp</sub>h<sub>as</sub>i<sub>zes c</sub>h<sub>arac</sub>t<sub>er</sub> d<sub>es</sub>i<sub>gn, scene cons</sub>t<sub>ruc</sub>ti<sub>on, ren</sub>d<sub>er</sub>i<sub>ng cons</sub>i<sub>s</sub>t<sub>ency, narra</sub>ti<sub>ve</sub> <sub>se</sub>tti<sub>ng, an</sub>d <sub>s</sub>t<sub>y</sub>li<sub>za</sub>ti<sub>on.</sub> M<sub>u</sub>lti<sub>p</sub>l<sub>e</sub> MLLM<sub>s genera</sub>t<sub>e s</sub>t<sub>ruc</sub>t<sub>ure</sub>d <sub>anno</sub>t<sub>a</sub>ti<sub>ons, w</sub>hi<sub>c</sub>h <sub>are conver</sub>t<sub>e</sub>d i<sub>n</sub>t<sub>o mu</sub>lti<sub>-</sub>t<sub>urn</sub> i<sub>ns</sub>t<sub>ruc</sub>ti<sub>on–</sub> <sub>response conversa</sub>ti<sub>ons cover</sub>i<sub>ng</sub> d<sub>escr</sub>i<sub>p</sub>ti<sub>on, eva</sub>l<sub>ua</sub>ti<sub>on,</sub> i<sub>n-</sub> t<sub>erpre</sub>t<sub>a</sub>ti<sub>on,</sub> t<sub>ec</sub>h<sub>n</sub>i<sub>ca</sub>l <sub>ana</sub>l<sub>ys</sub>i<sub>s, an</sub>d <sub>cons</sub>t<sub>ruc</sub>ti<sub>ve</sub> i<sub>mprove-</sub> ment.

A<sub>n</sub> i<sub>n</sub>d<sub>epen</sub>d<sub>en</sub>t Cl<sub>au</sub>d<sub>e</sub> O<sub>pus</sub> 5 <sub>ver</sub>ifi<sub>er screens genera</sub>t<sub>e</sub>d <sub>pa</sub>i<sub>rs</sub> f<sub>or</sub> i<sub>ns</sub>t<sub>ruc</sub>ti<sub>on</sub> <sub>cons</sub>i<sub>s</sub>t<sub>ency,</sub> <sub>v</sub>i<sub>sua</sub>l <sub>groun</sub>di<sub>ng,</sub> <sub>an</sub>d <sub>re-</sub> <sub>sponse qua</sub>lit<sub>y.</sub> A <sub>s</sub>t<sub>ra</sub>tifi<sub>e</sub>d <sub>au</sub>dit <sub>o</sub>f1<sub>,</sub>000 <sub>re</sub>t<sub>a</sub>i<sub>ne</sub>d <sub>pa</sub>i<sub>rs y</sub>i<sub>e</sub>ld<sub>s</sub> 95<sub>.</sub>3% <sub>accep</sub>t<sub>ance</sub> <sub>an</sub>d 97% <sub>raw</sub> <sub>agreemen</sub>t<sub>;</sub> f<sub>u</sub>ll <sub>cr</sub>it<sub>er</sub>i<sub>a</sub> <sub>an</sub>d <sub>s</sub>t<sub>a</sub>ti<sub>s</sub>ti<sub>cs appear</sub> i<sub>n</sub> th<sub>e supp</sub>l<sub>emen</sub>t<sub>.</sub>

C<sub>on</sub>textCa<sub>nv</sub>a<sub>s</sub> C<sub>o</sub>n<sub>s</sub>tr<sub>uc</sub>ti<sub>o</sub>n <sub>a</sub>nd R<sub>ev</sub>i<sub>ew.</sub> A<sub>s s</sub>h<sub>own</sub> in Fi<sub>g</sub>. 3(c), ContextCanvas is constructed from a selected <sub>su</sub>b<sub>se</sub>t <sub>o</sub>f th<sub>e s</sub>h<sub>are</sub>d i<sub>mage poo</sub>l<sub>.</sub> F<sub>our</sub> PhD<sub>-</sub>l<sub>eve</sub>l <sub>researc</sub>h<sub>ers</sub> <sub>w</sub>ith i<sub>n</sub>t<sub>er</sub>di<sub>sc</sub>i<sub>p</sub>li<sub>nary</sub> b<sub>ac</sub>k<sub>groun</sub>d<sub>s</sub> d<sub>eve</sub>l<sub>ope</sub>d <sub>an</sub>d <sub>rev</sub>i<sub>ewe</sub>d 1<sub>,</sub>060 <sub>can</sub>did<sub>a</sub>t<sub>e</sub> <sub>cases.</sub> E<sub>ac</sub>h <sub>p</sub>l<sub>aces</sub> <sub>an</sub> i<sub>mage</sub> i<sub>n</sub> <sub>a</sub> <sub>p</sub>l<sub>aus</sub>i<sub>-</sub> bl<sub>e use scenar</sub>i<sub>o—suc</sub>h <sub>as ex</sub>hibiti<sub>on, pu</sub>bli<sub>ca</sub>ti<sub>on, a</sub>d<sub>ver</sub>ti<sub>s-</sub> i<sub>ng,</sub> <sub>e</sub>d<sub>uca</sub>ti<sub>on,</sub> <sub>commemora</sub>ti<sub>on,</sub> <sub>or</sub> <sub>pu</sub>bli<sub>c</sub> <sub>commun</sub>i<sub>ca</sub>ti<sub>on—</sub> <sub>an</sub>d <sub>as</sub>k<sub>s w</sub>h<sub>e</sub>th<sub>er</sub> it i<sub>s an appropr</sub>i<sub>a</sub>t<sub>e v</sub>i<sub>sua</sub>l <sub>c</sub>h<sub>o</sub>i<sub>ce</sub> f<sub>or</sub> th<sub>e</sub> <sub>s</sub>t<sub>a</sub>t<sub>e</sub>d <sub>purpose, au</sub>di<sub>ence, an</sub>d <sub>con</sub>t<sub>ex</sub>t<sub>.</sub> S<sub>ome cases</sub> i<sub>nc</sub>l<sub>u</sub>d<sub>e</sub> <sub>a cura</sub>t<sub>or- or</sub> d<sub>es</sub>i<sub>gner-propose</sub>d <sub>assessmen</sub>t th<sub>a</sub>t th<sub>e mo</sub>d<sub>e</sub>l <sub>mus</sub>t i<sub>n</sub>d<sub>epen</sub>d<sub>en</sub>tl<sub>y eva</sub>l<sub>ua</sub>t<sub>e.</sub>

A <sub>case</sub> i<sub>s</sub> <sub>re</sub>t<sub>a</sub>i<sub>ne</sub>d <sub>on</sub>l<sub>y</sub> <sub>w</sub>h<sub>en</sub> it <sub>sa</sub>ti<sub>s</sub>fi<sub>es</sub> <sub>seven</sub> <sub>prespec</sub>ifi<sub>e</sub>d criteria: (1) aesthetic centrality, requiring a judgment about the image as a visual choice; (2) context dependence, such that intrinsic quality alone is insuficient; (3) visual necessity, <sub>suc</sub>h th<sub>a</sub>t th<sub>e</sub> t<sub>as</sub>k <sub>canno</sub>t b<sub>e re</sub>d<sub>uce</sub>d t<sub>o</sub> t<sub>ex</sub>t<sub>-on</sub>l<sub>y</sub> f<sub>ac</sub>t<sub>ua</sub>l <sub>or</sub> cultural knowledge; (4) decisive evidence, such that relevant <sub>cu</sub>lt<sub>ura</sub>l<sub>,</sub> hi<sub>s</sub>t<sub>or</sub>i<sub>ca</sub>l<sub>, or narra</sub>ti<sub>ve ev</sub>id<sub>ence ma</sub>t<sub>er</sub>i<sub>a</sub>ll<sub>y a</sub>f<sub>ec</sub>t<sub>s</sub> the answer; (5) scenario naturalness, requiring a plausible design or communication use; (6) answerability, requiring a <sub>un</sub>i<sub>que c</sub>l<sub>ose</sub>d<sub>-</sub>f<sub>orm answer</sub> f<sub>rom percep</sub>tibl<sub>e ev</sub>id<sub>ence; an</sub>d (7) anti-shortcut validity, preventing wording, option length, <sub>or</sub> l<sub>a</sub>b<sub>e</sub>l <sub>pr</sub>i<sub>ors</sub> f<sub>rom revea</sub>li<sub>ng</sub> th<sub>e answer.</sub>

<table><tr><td>Task</td><td>Metric Family</td><td>Metrics</td></tr><tr><td rowspan="4"></td><td>CRITIQUE Lexical overlap</td><td>BLEU, ROUGE-L, METEOR</td></tr><tr><td>Semantic similarity</td><td>BERT-F1, SBERT-Cos</td></tr><tr><td>Image-text alignment</td><td>CLIPScore</td></tr><tr><td></td><td>Descriptive statistics Avg. Length, Avg. Dim. Hits, Top-3 Dimensions</td></tr><tr><td rowspan="2"></td><td>CoNTEXT Prediction quality</td><td>Accuracy, Macro-F1</td></tr><tr><td>Prediction tendency</td><td>Yes Rate</td></tr></table>

T<sub>a</sub>bl<sub>e</sub> 1<sub>:</sub> M<sub>e</sub>t<sub>r</sub>i<sub>cs use</sub>d f<sub>or</sub> th<sub>e</sub> t<sub>wo eva</sub>l<sub>ua</sub>ti<sub>on</sub> t<sub>as</sub>k<sub>s.</sub>

All <sub>can</sub>did<sub>a</sub>t<sub>es were rev</sub>i<sub>ewe</sub>d <sub>aga</sub>i<sub>ns</sub>t th<sub>ese cr</sub>it<sub>er</sub>i<sub>a, w</sub>ith borderline cases resolved throu<sub>g</sub>h discussion. Qwen3.7-Plus <sub>was use</sub>d <sub>on</sub>l<sub>y as a</sub> d<sub>eve</sub>l<sub>opmen</sub>t<sub>-</sub>ti<sub>me pro</sub>b<sub>e</sub> f<sub>or</sub> t<sub>r</sub>i<sub>v</sub>i<sub>a</sub>l<sub>, am-</sub> bi<sub>guous, or s</sub>h<sub>or</sub>t<sub>cu</sub>t<sub>-prone cases; mo</sub>d<sub>e</sub>l f<sub>a</sub>il<sub>ure a</sub>l<sub>one never</sub> d<sub>e</sub>t<sub>erm</sub>i<sub>ne</sub>d <sub>re</sub>t<sub>en</sub>ti<sub>on.</sub> All i<sub>nc</sub>l<sub>us</sub>i<sub>on</sub> d<sub>ec</sub>i<sub>s</sub>i<sub>ons, go</sub>ld l<sub>a</sub>b<sub>e</sub>l<sub>s,</sub> <sub>ra</sub>ti<sub>ona</sub>l<sub>es, an</sub>d <sub>source recor</sub>d<sub>s were</sub> fi<sub>na</sub>li<sub>ze</sub>d <sub>so</sub>l<sub>e</sub>l<sub>y</sub> b<sub>y</sub> h<sub>u-</sub> <sub>man rev</sub>i<sub>ewers, an</sub>d it<sub>s repor</sub>t<sub>e</sub>d <sub>score</sub> i<sub>s</sub> th<sub>ere</sub>f<sub>ore mar</sub>k<sub>e</sub>d <sub>as</sub> <sub>a</sub> d<sub>eve</sub>l<sub>opmen</sub>t<sub>-mo</sub>d<sub>e</sub>l di<sub>agnos</sub>ti<sub>c.</sub>

The final benchmark contains 301 cases (28.40%): 300 <sub>s</sub>i<sub>ng</sub>l<sub>e-</sub>i<sub>mage</sub> <sub>an</sub>d <sub>one</sub> <sub>pa</sub>i<sub>re</sub>d<sub>-</sub>i<sub>mage</sub> <sub>case,</sub> <sub>cover</sub>i<sub>ng</sub> 302 i<sub>m-</sub> <sub>ages,</sub> 291 bi<sub>nary</sub> <sub>ques</sub>ti<sub>ons,</sub> <sub>an</sub>d 10 th<sub>ree-way</sub> <sub>ques</sub>ti<sub>ons.</sub> E<sub>ac</sub>h i<sub>nc</sub>l<sub>u</sub>d<sub>es</sub> th<sub>e</sub> <sub>v</sub>i<sub>sua</sub>l i<sub>npu</sub>t<sub>,</sub> <sub>use</sub> <sub>scenar</sub>i<sub>o,</sub> <sub>con</sub>t<sub>ex</sub>t<sub>,</sub> <sub>answer</sub> <sub>op-</sub> ti<sub>ons,</sub> <sub>go</sub>ld l<sub>a</sub>b<sub>e</sub>l<sub>,</sub> <sub>source-groun</sub>d<sub>e</sub>d <sub>ra</sub>ti<sub>ona</sub>l<sub>e,</sub> <sub>an</sub>d <sub>provenance.</sub> Th<sub>e</sub> <sub>cases</sub> <sub>span</sub> <sub>pa</sub>i<sub>n</sub>ti<sub>ng,</sub> <sub>scu</sub>l<sub>p</sub>t<sub>ure,</sub> ill<sub>us</sub>t<sub>ra</sub>ti<sub>on</sub> <sub>an</sub>d <sub>com</sub>i<sub>cs,</sub> <sub>an</sub>i<sub>ma</sub>ti<sub>on an</sub>d di<sub>g</sub>it<sub>a</sub>l i<sub>magery, p</sub>h<sub>o</sub>t<sub>ograp</sub>h<sub>y, an</sub>d fil<sub>m.</sub>

## C E<sub>va</sub>l<sub>ua</sub>ti<sub>on</sub> P<sub>ro</sub>t<sub>oco</sub>l

W<sub>e</sub> <sub>eva</sub>l<sub>ua</sub>t<sub>e</sub> <sub>cr</sub>iti<sub>que</sub> <sub>genera</sub>ti<sub>on</sub> <sub>on</sub> C<sub>ritique</sub>C<sub>anvas</sub> <sub>an</sub>d <sub>con</sub>t<sub>ex</sub>t<sub>ua</sub>l <sub>su</sub>it<sub>a</sub>bilit<sub>y on</sub> C<sub>ontext</sub>C<sub>anvas.</sub> M<sub>o</sub>d<sub>e</sub>l<sub>s rece</sub>i<sub>ve</sub> id<sub>en</sub>ti<sub>ca</sub>l i<sub>npu</sub>t<sub>s an</sub>d i<sub>ns</sub>t<sub>ruc</sub>ti<sub>ons w</sub>ith<sub>ou</sub>t <sub>we</sub>b <sub>searc</sub>h<sub>, re-</sub> t<sub>r</sub>i<sub>eva</sub>l<sub>, or aux</sub>ili<sub>ary recogn</sub>iti<sub>on</sub> t<sub>oo</sub>l<sub>s;</sub> T<sub>a</sub>bl<sub>e</sub> 1 <sub>summar</sub>i<sub>zes</sub> th<sub>e me</sub>t<sub>r</sub>i<sub>cs.</sub>

L<sub>o</sub>n<sub>g-</sub>f<sub>o</sub>rm A<sub>es</sub>th<sub>e</sub>ti<sub>c</sub> Criti<sub>que</sub> G<sub>e</sub>n<sub>e</sub>r<sub>a</sub>ti<sub>o</sub>n<sub>.</sub> W<sub>e eva</sub>l<sub>u-</sub> <sub>a</sub>t<sub>e mo</sub>d<sub>e</sub>l<sub>s on</sub> 6<sub>,</sub>000 <sub>open-en</sub>d<sub>e</sub>d <sub>samp</sub>l<sub>es</sub> b<sub>a</sub>l<sub>ance</sub>d <sub>across</sub> <sub>p</sub><sup>h</sup>oto<sub>g</sub>ra<sub>p</sub><sup>h</sup><sub>y</sub>, <sub>p</sub>a<sup>i</sup>nt<sup>i</sup>n<sub>g</sub>, an<sup>d</sup> v<sup>i</sup>rtua<sup>l</sup> <sup>i</sup>ma<sub>g</sub>er<sub>y</sub>, re<sub>q</sub>u<sup>i</sup>r<sup>i</sup>n<sub>g</sub> aesth<sub>e</sub>ti<sub>c ana</sub>l<sub>ys</sub>i<sub>s or</sub> i<sub>mage-groun</sub>d<sub>e</sub>d i<sub>mprovemen</sub>t <sub>sugges</sub>ti<sub>ons.</sub> I<sub>mage-</sub>l<sub>eve</sub>l <sub>sp</sub>lit<sub>s</sub> k<sub>eep re</sub>l<sub>a</sub>t<sub>e</sub>d <sub>conversa</sub>ti<sub>ons</sub> t<sub>oge</sub>th<sub>er.</sub>

B<sub>ecause</sub> <sub>va</sub>lid <sub>cr</sub>iti<sub>ques</sub> <sub>may</sub> dif<sub>er</sub> <sub>su</sub>b<sub>s</sub>t<sub>an</sub>ti<sub>a</sub>ll<sub>y</sub> i<sub>n</sub> <sub>wor</sub>d<sub>-</sub> in<sub>g</sub>, we re<sub>p</sub>ort com<sub>p</sub>lementar<sub>y</sub> metric families. BLEU (Pa-<sub>p</sub>ineni et al. 2002), ROUGE-L (Lin 2004), and METEOR (Banerjee and Lavie 2005) measure lexical overla<sub>p</sub>; BERT-F1 (Zhan<sub>g</sub> et al. 2020) and SBERT-Cos (Reimers and Gurev<sub>y</sub>ch 2019) measure token- and sentence-level semantic similarit<sub>y</sub>; and CLIPScore (Hessel et al. 2021) measures image–text relevance. These metrics are interpreted jointly <sub>ra</sub>th<sub>er</sub> th<sub>an as comp</sub>l<sub>e</sub>t<sub>e measures o</sub>f <sub>aes</sub>th<sub>e</sub>ti<sub>c reason</sub>i<sub>ng.</sub>

<table><tr><td>Model</td><td>Acc.</td><td>Macro-F1</td><td>Yes (%)</td></tr><tr><td colspan="4">Closed-source MLLMs</td></tr><tr><td> Claude Opus 5</td><td>91.69</td><td>90.45</td><td>30.24</td></tr><tr><td>Gemini 3.1 Pro</td><td>85.71</td><td>83.06</td><td>26.12</td></tr><tr><td>GPT-5.5</td><td>83.06</td><td>80.13</td><td>27.49</td></tr><tr><td>Claude Opus 4.6</td><td>72.76</td><td>67.96</td><td>27.15</td></tr><tr><td>GPT-5.2</td><td>71.43</td><td>64.51</td><td>20.96</td></tr><tr><td>G GLM-5V-Turbo</td><td>57.81</td><td>55.22</td><td>43.64</td></tr><tr><td>女 Qwen3.7-Plus†</td><td>51.83</td><td>48.39</td><td>41.58</td></tr><tr><td>Ø Grok 4.20</td><td>44.19</td><td>39.09</td><td>37.80</td></tr><tr><td colspan="4">Open-weight General MLLMs</td></tr><tr><td>Qwen3-VL-Instruct (Bai et al. 2025)</td><td>42.19</td><td>38.74</td><td>44.67</td></tr><tr><td>ov InternVL3 (Zhu et al. 2025)</td><td>23.26</td><td>21.65</td><td>62.20</td></tr><tr><td>+. mPLUG-Owl2 (Ye et al. 2024)</td><td>20.27</td><td>17.43</td><td>50.17</td></tr><tr><td>LLaVA-OneVision (An et al. 2025)</td><td>19.93</td><td>18.20</td><td>71.13</td></tr><tr><td>LLaVA-v1.5 (Liu et al. 2024)</td><td>12.62</td><td>10.29</td><td>60.48</td></tr><tr><td colspan="4">Aesthetic-specific Models</td></tr><tr><td>ArtQuant-APDD (Liu et al. 2026)</td><td>29.57</td><td>29.50</td><td>59.11</td></tr><tr><td>ArtiMuse (Cao et al. 2025b)</td><td>19.60</td><td>17.87</td><td>62.54</td></tr><tr><td>Q-SiT (Zhang et al. 2025)</td><td>19.60</td><td>16.65</td><td>76.98</td></tr><tr><td>AesExpert (Huang et al. 2024a)</td><td>18.27</td><td>16.30</td><td>66.67</td></tr></table>

Table 2: Contextual aesthetic suitabilit<sub>y</sub> results. Acc. is exact-match accurac<sub>y</sub> on all 301 cases. Macro-F1 and Yes (%) are com<sub>p</sub>uted on the 291 binar<sub>y</sub> cases; the <sub>g</sub>old Yes rate is 38.14%. Best Acc. and Macro-F1 are bolded. † denotes the develo<sub>p</sub>ment <sub>mo</sub>d<sub>e</sub>l <sub>use</sub>d f<sub>or</sub> difi<sub>cu</sub>lt<sub>y screen</sub>i<sub>ng.</sub>

T<sub>o</sub> f<sub>ur</sub>th<sub>er</sub> <sub>c</sub>h<sub>arac</sub>t<sub>er</sub>i<sub>ze</sub> <sub>response</sub> b<sub>e</sub>h<sub>av</sub>i<sub>or,</sub> <sub>we</sub> <sub>repor</sub>t <sub>aver-</sub> <sub>age response</sub> l<sub>eng</sub>th<sub>, average aes</sub>th<sub>e</sub>ti<sub>c-</sub>di<sub>mens</sub>i<sub>on coverage,</sub> <sub>an</sub>d th<sub>e</sub> th<sub>ree mos</sub>t f<sub>requen</sub>tl<sub>y a</sub>dd<sub>resse</sub>d di<sub>mens</sub>i<sub>ons</sub> i<sub>n</sub> th<sub>e</sub> su<sub>pp</sub>lementar<sub>y</sub> material. S<sub>p</sub>ecificall<sub>y</sub>, let D denote the <sub>p</sub>red<sub>e</sub>fi<sub>ne</sub>d <sub>se</sub>t <sub>o</sub>f <sub>aes</sub>th<sub>e</sub>ti<sub>c</sub> di<sub>mens</sub>i<sub>ons, an</sub>d l<sub>e</sub>t $m ( \hat { r } _ { i } , d )$ i<sub>n</sub>di<sub>ca</sub>t<sub>e</sub> <sub>w</sub>h<sub>e</sub>th<sub>er response</sub> $\boldsymbol { { \hat { r } } } _ { i }$ explicitly addresses dimension d. The <sub>average</sub> <sub>num</sub>b<sub>er</sub> <sub>o</sub>f di<sub>mens</sub>i<sub>ons</sub> <sub>a</sub>dd<sub>resse</sub>d <sub>per</sub> <sub>response</sub> i<sub>s</sub>

$$
\mathrm { D i m H i t s } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \sum _ { d \in \mathcal { D } } m ( \hat { \boldsymbol { r } } _ { i } , d ) .\tag{1}
$$

Thi<sub>s s</sub>t<sub>a</sub>ti<sub>s</sub>ti<sub>c measures</sub> th<sub>e</sub> b<sub>rea</sub>dth <sub>o</sub>f <sub>exp</sub>li<sub>c</sub>it <sub>aes</sub>th<sub>e</sub>ti<sub>c</sub> <sub>coverage ra</sub>th<sub>er</sub> th<sub>an</sub> it<sub>s correc</sub>t<sub>ness: a</sub> hi<sub>g</sub>h<sub>er va</sub>l<sub>ue</sub> d<sub>oes no</sub>t <sub>necessar</sub>il<sub>y</sub> i<sub>n</sub>di<sub>ca</sub>t<sub>e</sub> th<sub>a</sub>t th<sub>e</sub> id<sub>en</sub>tifi<sub>e</sub>d di<sub>mens</sub>i<sub>ons are re</sub>l<sub>e-</sub> <sub>van</sub>t<sub>, accura</sub>t<sub>e, or we</sub>ll <sub>groun</sub>d<sub>e</sub>d<sub>.</sub>

C<sub>o</sub>nt<sub>e</sub>xt<sub>ua</sub>l A<sub>es</sub>th<sub>e</sub>ti<sub>c</sub> S<sub>u</sub>it<sub>a</sub>bilit<sub>y</sub> J<sub>u</sub>d<sub>g</sub>m<sub>e</sub>nt<sub>.</sub> F<sub>or eac</sub>h C<sub>ontext</sub>C<sub>anvas case,</sub> th<sub>e mo</sub>d<sub>e</sub>l <sub>rece</sub>i<sub>ves an</sub> i<sub>mage, a re-</sub> <sub>a</sub>li<sub>s</sub>ti<sub>c</sub> <sub>use</sub> <sub>scenar</sub>i<sub>o,</sub> <sub>an</sub>d fi<sub>xe</sub>d <sub>answer</sub> <sub>op</sub>ti<sub>ons,</sub> <sub>an</sub>d <sub>re</sub>t<sub>urns</sub> <sub>a</sub> <sub>s</sub>i<sub>ng</sub>l<sub>e op</sub>ti<sub>on</sub> l<sub>a</sub>b<sub>e</sub>l<sub>.</sub> E<sub>xp</sub>l<sub>ana</sub>ti<sub>ons are re</sub>t<sub>a</sub>i<sub>ne</sub>d f<sub>or qua</sub>lit<sub>a</sub>ti<sub>ve</sub> <sub>ana</sub>l<sub>ys</sub>i<sub>s</sub> b<sub>u</sub>t d<sub>o no</sub>t <sub>a</sub>f<sub>ec</sub>t th<sub>e pr</sub>i<sub>mary score.</sub> A <sub>pre</sub>di<sub>c</sub>ti<sub>on</sub> i<sub>s correc</sub>t <sub>on</sub>l<sub>y</sub> if <sub>a un</sub>i<sub>que op</sub>ti<sub>on</sub> i<sub>s parse</sub>d <sub>an</sub>d <sub>ma</sub>t<sub>c</sub>h<sub>es</sub> th<sub>e</sub> <sub>go</sub>ld <sub>answer; m</sub>i<sub>ss</sub>i<sub>ng, con</sub>fli<sub>c</sub>ti<sub>ng, or unparsa</sub>bl<sub>e responses</sub> <sub>are coun</sub>t<sub>e</sub>d <sub>as</sub> i<sub>ncorrec</sub>t<sub>.</sub>

W<sub>e</sub> r<sub>epo</sub>rt <sub>exac</sub>t<sub>-</sub>m<sub>a</sub>t<sub>c</sub>h <sub>accu</sub>r<sub>acy</sub> <sub>ove</sub>r <sub>a</sub>ll 301 <sub>cases.</sub> M<sub>ac</sub>r<sub>o-</sub> F1 <sub>an</sub>d <sub>pre</sub>di<sub>c</sub>t<sub>e</sub>d Y<sub>es</sub> R<sub>a</sub>t<sub>e are compu</sub>t<sub>e</sub>d <sub>on</sub> th<sub>e</sub> 291 bi<sub>-</sub> nary cases after mapping the options to canonical Yes/No l<sub>a</sub>b<sub>e</sub>l<sub>s;</sub> th<sub>e rema</sub>i<sub>n</sub>i<sub>ng</sub> 10 th<sub>ree-way cases are</sub> i<sub>nc</sub>l<sub>u</sub>d<sub>e</sub>d <sub>on</sub>l<sub>y</sub> i<sub>n accuracy.</sub> Th<sub>e go</sub>ld Y<sub>es</sub> R<sub>a</sub>t<sub>e</sub> i<sub>s</sub> 38<sub>.</sub>14%<sub>, aga</sub>i<sub>ns</sub>t <sub>w</sub>hi<sub>c</sub>h th<sub>e</sub> <sub>pre</sub>di<sub>c</sub>t<sub>e</sub>d <sub>ra</sub>t<sub>e</sub> i<sub>n</sub>di<sub>ca</sub>t<sub>es</sub> <sub>a</sub> <sub>mo</sub>d<sub>e</sub>l’<sub>s</sub> t<sub>en</sub>d<sub>ency</sub> t<sub>owar</sub>d <sub>over-</sub> acceptance or over-rejection.

## IV Ex<sub>pe</sub>rim<sub>e</sub>nt<sub>s</sub>

## A E<sub>xpe</sub>rim<sub>e</sub>nt<sub>a</sub>l S<sub>e</sub>t<sub>up</sub>

E<sub>va</sub>l<sub>ua</sub>t<sub>e</sub>d M<sub>o</sub>d<sub>e</sub>l<sub>s.</sub> W<sub>e compare c</sub>l<sub>ose</sub>d<sub>-source</sub> f<sub>ron</sub>ti<sub>er</sub> MLLM<sub>s, ope</sub>n<sub>-we</sub>i<sub>g</sub>ht <sub>ge</sub>n<sub>e</sub>r<sub>a</sub>l MLLM<sub>s, a</sub>nd <sub>aes</sub>th<sub>e</sub>ti<sub>c-</sub> <sub>spec</sub>ifi<sub>c</sub> <sub>mo</sub>d<sub>e</sub>l<sub>s,</sub> <sub>separa</sub>ti<sub>ng</sub> <sub>genera</sub>l <sub>v</sub>i<sub>sua</sub>l<sub>-</sub>l<sub>anguage</sub> <sub>capa</sub>bil<sub>-</sub> it<sub>y</sub> f<sub>rom spec</sub>i<sub>a</sub>li<sub>za</sub>ti<sub>on</sub> f<sub>or aes</sub>th<sub>e</sub>ti<sub>c assessmen</sub>t <sub>or cr</sub>iti<sub>que.</sub> Exact model variants are listed in Tables 2 and 3. Qwen3- VL d<sub>eno</sub>t<sub>es a var</sub>i<sub>an</sub>t <sub>a</sub>d<sub>ap</sub>t<sub>e</sub>d <sub>on</sub> C<sub>ritique</sub>C<sub>anvas an</sub>d <sub>eva</sub>l<sub>ua</sub>t<sub>e</sub>d <sub>on</sub>l<sub>y on cr</sub>iti<sub>que genera</sub>ti<sub>on.</sub>

Im<sub>p</sub>l<sub>e</sub>m<sub>e</sub>nt<sub>a</sub>ti<sub>o</sub>n D<sub>e</sub>t<sub>a</sub>il<sub>s.</sub> All <sub>mo</sub>d<sub>e</sub>l<sub>s</sub> f<sub>o</sub>ll<sub>ow</sub> th<sub>e pro</sub>t<sub>oco</sub>l i<sub>n</sub> S<sub>ec</sub>ti<sub>on</sub> III<sub>.</sub>C<sub>.</sub> L<sub>oca</sub>ll<sub>y</sub> d<sub>ep</sub>l<sub>oye</sub>d <sub>mo</sub>d<sub>e</sub>l<sub>s rece</sub>i<sub>ve</sub> i<sub>mages</sub> resized or padded to 448 × 448 pixels. We use deterministic d<sub>eco</sub>di<sub>ng w</sub>h<sub>enever suppor</sub>t<sub>e</sub>d <sub>an</sub>d <sub>a max</sub>i<sub>mum o</sub>f 256 t<sub>o</sub>k<sub>ens</sub> f<sub>or cr</sub>iti<sub>que genera</sub>ti<sub>on; mo</sub>d<sub>e</sub>l<sub>-spec</sub>ifi<sub>c se</sub>tti<sub>ngs are repor</sub>t<sub>e</sub>d i<sub>n</sub> th<sub>e</sub> <sub>supp</sub>l<sub>emen</sub>t<sub>ary</sub> <sub>ma</sub>t<sub>er</sub>i<sub>a</sub>l<sub>.</sub>

## B M<sub>a</sub>i<sub>n</sub> R<sub>esu</sub>lt<sub>s</sub>

C<sub>o</sub>nt<sub>e</sub>xt<sub>ua</sub>l A<sub>es</sub>th<sub>e</sub>ti<sub>c</sub> S<sub>u</sub>it<sub>a</sub>bilit<sub>y</sub> J<sub>u</sub>d<sub>g</sub>m<sub>e</sub>nt T<sub>a</sub>bl<sub>e</sub> 2 <sub>re-</sub> <sub>por</sub>t<sub>s resu</sub>lt<sub>s on</sub> C<sub>ontext</sub>C<sub>anvas.</sub> Th<sub>e</sub> b<sub>enc</sub>h<sub>mar</sub>k <sub>c</sub>l<sub>ear</sub>l<sub>y</sub> <sub>separa</sub>t<sub>es</sub> <sub>mo</sub>d<sub>e</sub>l <sub>capa</sub>bilit<sub>y</sub> l<sub>eve</sub>l<sub>s.</sub> A<sub>ccuracy</sub> <sub>among</sub> <sub>c</sub>l<sub>ose</sub>d<sub>-</sub> source MLLMs ranges from 44.19% to 91.69%, while the stron<sub>g</sub>est evaluated open-wei<sub>g</sub>ht model, Qwen3-VL-Instruct, reaches only 42.19%. Within the Claude and GPT families, <sub>newer var</sub>i<sub>an</sub>t<sub>s su</sub>b<sub>s</sub>t<sub>an</sub>ti<sub>a</sub>ll<sub>y ou</sub>t<sub>per</sub>f<sub>orm</sub> th<sub>e</sub>i<sub>r pre</sub>d<sub>ecessors:</sub> Claude Opus 5 improves over Claude Opus 4.6 by 18.93 accuracy points and 22.49 Macro-F1 points, while GPT-5.5 improves over GPT-5.2 by 11.63 and 15.62 points, respectively.

<table><tr><td>Model</td><td>BLEU</td><td>ROUGE-L</td><td>METEOR</td><td>BERT-F1</td><td>SBERT-Cos</td><td>CLIPScore</td></tr><tr><td colspan="7">Closed-source MLLMs</td></tr><tr><td>Gemini 3.1 Pro</td><td>7.15</td><td>0.250</td><td>0.281</td><td>0.699</td><td>0.848</td><td>0.303</td></tr><tr><td>GPT-5.2</td><td>2.84</td><td>0.210</td><td>0.209</td><td>0.669</td><td>0.818</td><td>0.301</td></tr><tr><td colspan="7">Open-weight General MLLMs</td></tr><tr><td>GLM-4.6-Flash (GLM-V Team et al. 2026)</td><td>5.50</td><td>0.241</td><td>0.273</td><td>0.676</td><td>0.844</td><td>0.317</td></tr><tr><td>LLaVA-OneVision(An et al. 2025)</td><td>10.32</td><td>0.285</td><td>0.275</td><td>0.724</td><td>0.839</td><td>0.319</td></tr><tr><td>InternVL (Wang et al. 2025) </td><td>9.15</td><td>0.272</td><td>0.280</td><td>0.716</td><td>0.829</td><td>0.300</td></tr><tr><td>Qwen3-VL (Bai et al. 2025)</td><td>5.12</td><td>0.232</td><td>0.274</td><td>0.684</td><td>0.827</td><td>0.313</td></tr><tr><td>Qwen3-VLFT</td><td>12.23</td><td>0.297</td><td>0.296</td><td>0.722</td><td>0.787</td><td>0.285</td></tr><tr><td colspan="7">Aesthetic-specific Models</td></tr><tr><td>ArtQuant-APDD (Liu et al. 2026)</td><td>10.68</td><td>0.280</td><td>0.277</td><td>0.716</td><td>0.818</td><td>0.297</td></tr><tr><td>ArtiMuse (Cao et al. 2025b)</td><td>7.98</td><td>0.271</td><td>0.239</td><td>0.718</td><td>0.826</td><td>0.312</td></tr><tr><td>UniPercept (Cao et al. 2025a)</td><td>5.12</td><td>0.235</td><td>0.200</td><td>0.693</td><td>0.823</td><td>0.318</td></tr><tr><td>AesExpert (Huang et al. 2024a)</td><td>2.08</td><td>0.198</td><td>0.128</td><td>0.651</td><td>0.609</td><td>0.262</td></tr><tr><td>Q-SiT (Zhang et al. 2025)</td><td>1.31</td><td>0.188</td><td>0.108</td><td>0.635</td><td>0.508</td><td>0.261</td></tr></table>

Table 3: Quantitative evaluation of lon<sub>g</sub>-form aesthetic critique <sub>g</sub>eneration on CritiqueCanvas. The reported metrics assess <sub>comp</sub>l<sub>emen</sub>t<sub>ary aspec</sub>t<sub>s o</sub>f <sub>genera</sub>t<sub>e</sub>d <sub>cr</sub>iti<sub>ques,</sub> i<sub>nc</sub>l<sub>u</sub>di<sub>ng re</sub>f<sub>erence a</sub>li<sub>gnmen</sub>t th<sub>roug</sub>h l<sub>ex</sub>i<sub>ca</sub>l <sub>an</sub>d <sub>seman</sub>ti<sub>c s</sub>i<sub>m</sub>il<sub>ar</sub>it<sub>y, an</sub>d i<sub>mage–</sub>t<sub>ex</sub>t <sub>re</sub>l<sub>evance.</sub>

Th<sub>ese w</sub>ithi<sub>n-</sub>f<sub>am</sub>il<sub>y ga</sub>i<sub>ns are cons</sub>i<sub>s</sub>t<sub>en</sub>t <sub>w</sub>ith i<sub>mprove</sub>d <sub>con-</sub> textual aesthetic judgment in newer frontier models.

M<sub>os</sub>t <sub>no</sub>t<sub>a</sub>bl<sub>y, a</sub>ll <sub>eva</sub>l<sub>ua</sub>t<sub>e</sub>d <sub>aes</sub>th<sub>e</sub>ti<sub>c-spec</sub>ifi<sub>c mo</sub>d<sub>e</sub>l<sub>s</sub> achieve below 30% accuracy. ArtQuant-APDD, the strongest model in this group, obtains 29.57%, compared with 42.19% for Qwen3-VL-Instruct and 91.69% for Claude Opus 5. ArtiMuse, Q-SiT, and AesExpert achieve only 19.60%, 19.60%, and 18.27%, respectively. Thus, specialization for aesthetic <sub>scor</sub>i<sub>ng, percep</sub>ti<sub>on, or cr</sub>iti<sub>que</sub> d<sub>oes no</sub>t <sub>re</sub>li<sub>a</sub>bl<sub>y</sub> t<sub>rans</sub>f<sub>er</sub> to judging suitability under cultural, communicative, or d<sub>oma</sub>i<sub>n-spec</sub>ifi<sub>c cons</sub>t<sub>ra</sub>i<sub>n</sub>t<sub>s.</sub>

Th<sub>e</sub> Y<sub>es-ra</sub>t<sub>e co</sub>l<sub>umn</sub> f<sub>ur</sub>th<sub>er revea</sub>l<sub>s a s</sub>h<sub>are</sub>d <sub>pre</sub>di<sub>c</sub>ti<sub>on</sub> tendency. Against the gold rate of 38.14%, aesthetic-specific models predict Yes in 59.11%–76.98% of the binary cases, i<sub>n</sub>di<sub>ca</sub>ti<sub>ng pronounce</sub>d <sub>over-accep</sub>t<sub>ance</sub> d<sub>esp</sub>it<sub>e con</sub>t<sub>ex</sub>t<sub>ua</sub> <sub>m</sub>i<sub>sma</sub>t<sub>c</sub>h<sub>.</sub> Thi<sub>s pa</sub>tt<sub>ern</sub> i<sub>s cons</sub>i<sub>s</sub>t<sub>en</sub>t <sub>w</sub>ith <sub>aes</sub>th<sub>e</sub>ti<sub>c con</sub>t<sub>ex</sub>t bi<sub>as, a</sub>lth<sub>oug</sub>h <sub>s</sub>i<sub>m</sub>il<sub>ar marg</sub>i<sub>na</sub>l <sub>ra</sub>t<sub>es may ar</sub>i<sub>se</sub> f<sub>rom</sub> dif<sub>er-</sub> <sub>en</sub>t i<sub>ns</sub>t<sub>ance-</sub>l<sub>eve</sub>l f<sub>a</sub>il<sub>ures.</sub> W<sub>e exam</sub>i<sub>ne</sub> th<sub>ese</sub> f<sub>a</sub>il<sub>ure mo</sub>d<sub>es</sub> th<sub>roug</sub>h <sub>ca</sub>t<sub>egory-</sub>l<sub>eve</sub>l <sub>an</sub>d <sub>qua</sub>lit<sub>a</sub>ti<sub>ve ana</sub>l<sub>yses</sub> i<sub>n</sub> th<sub>e</sub> f<sub>o</sub>l<sub>-</sub> lowing section. Although the binary subset has a 61.86% always-No majority baseline, weaker models often fall below it b<sub>ecause</sub> th<sub>ey over-accep</sub>t <sub>v</sub>i<sub>sua</sub>ll<sub>y p</sub>l<sub>aus</sub>ibl<sub>e</sub> b<sub>u</sub>t <sub>con</sub>t<sub>ex</sub>t<sub>u-</sub> <sub>a</sub>ll<sub>y unsu</sub>it<sub>a</sub>bl<sub>e</sub> i<sub>mages.</sub> M<sub>eanw</sub>hil<sub>e,</sub> f<sub>ron</sub>ti<sub>er mo</sub>d<sub>e</sub>l<sub>s ac</sub>hi<sub>eve</sub> up to 91.69% accuracy on the full benchmark, suggesting that ContextCanvas rewards context-sensitive judgment rather than majority-label guessing.

L<sub>o</sub>n<sub>g</sub>-f<sub>o</sub>rm A<sub>es</sub>th<sub>e</sub>ti<sub>c</sub> Criti<sub>que</sub> G<sub>e</sub>n<sub>e</sub>r<sub>a</sub>ti<sub>o</sub>n T<sub>a</sub>bl<sub>e</sub> 3 <sub>re-</sub> <sub>por</sub>t<sub>s</sub> <sub>resu</sub>lt<sub>s</sub> <sub>on</sub> C<sub>ritique</sub>C<sub>anvas.</sub> N<sub>o</sub> <sub>mo</sub>d<sub>e</sub>l d<sub>om</sub>i<sub>na</sub>t<sub>es</sub> <sub>across a</sub>ll <sub>me</sub>t<sub>r</sub>i<sub>c</sub> f<sub>am</sub>ili<sub>es.</sub> G<sub>em</sub>i<sub>n</sub>i 3<sub>.</sub>1 P<sub>ro ac</sub>hi<sub>eves</sub> th<sub>e</sub> highest SBERT-Cos (0.848), while LLaVA-OneVision leads BERT-F1 (0.724) and CLIPScore (0.319). The adapted Qwen3-VL<sub>FT</sub> instead obtains the stron<sub>g</sub>est lexical scores: 12.23 BLEU, 0.297 ROUGE-L, and 0.296 METEOR. Compared with the base Qwen3-VL, these correspond to <sub>g</sub>ains of 7.11, 0.065, and 0.022, respectively. However, its SBERT-

Cos decreases from 0.827 to 0.787, and CLIPScore from 0.313 to 0.285, showing that adaptation improves reference-<sub>s</sub>t<sub>y</sub>l<sub>e ma</sub>t<sub>c</sub>hi<sub>ng w</sub>ith<sub>ou</sub>t <sub>un</sub>if<sub>orm ga</sub>i<sub>ns</sub> i<sub>n seman</sub>ti<sub>c s</sub>i<sub>m</sub>il<sub>ar</sub>it<sub>y</sub> or <sup>i</sup>ma<sub>g</sub>e <sub>g</sub>roun<sup>di</sup>n<sub>g</sub>.

A<sub>es</sub>th<sub>e</sub>ti<sub>c-spec</sub>ifi<sub>c mo</sub>d<sub>e</sub>l<sub>s rema</sub>i<sub>n compe</sub>titi<sub>ve on se</sub>l<sub>ec</sub>t<sub>e</sub>d critique metrics. ArtiMuse reaches 0.718 BERT-F1, close to the best score of 0.724, while UniPercept obtains a CLIP-Score of 0.318, only 0.001 below the overall best. ArtQuant also reaches 10.68 BLEU and 0.280 ROUGE-L, whereas AesExpert and Q-SiT perform weakl<sub>y</sub> across most metrics. O<sub>vera</sub>ll <sub>cr</sub>iti<sub>que per</sub>f<sub>ormance</sub> i<sub>s</sub> h<sub>e</sub>t<sub>erogeneous across</sub> l<sub>ex</sub>i<sub>ca</sub>l<sub>,</sub> <sub>seman</sub>ti<sub>c, an</sub>d i<sub>mage-groun</sub>di<sub>ng cr</sub>it<sub>er</sub>i<sub>a.</sub>

T<sub>oge</sub>th<sub>er,</sub> th<sub>e</sub> t<sub>wo</sub> t<sub>as</sub>k<sub>s</sub> <sub>revea</sub>l <sub>a</sub> <sub>gap</sub> b<sub>e</sub>t<sub>ween</sub> <sub>re</sub>f<sub>erence-</sub> <sub>a</sub>li<sub>gne</sub>d <sub>cr</sub>iti<sub>que</sub> <sub>genera</sub>ti<sub>on</sub> <sub>an</sub>d <sub>con</sub>t<sub>ex</sub>t<sub>ua</sub>ll<sub>y</sub> <sub>va</sub>lid <sub>aes</sub>th<sub>e</sub>ti<sub>c</sub> judgment. Additional response-level statistics are provided i<sub>n</sub> th<sub>e supp</sub>l<sub>emen</sub>t<sub>ary ma</sub>t<sub>er</sub>i<sub>a</sub>l<sub>.</sub>

## C An<sub>a</sub>l<sub>ys</sub>i<sub>s</sub>

B<sub>eyo</sub>nd <sub>agg</sub>r<sub>ega</sub>t<sub>e</sub> <sub>sco</sub>r<sub>es,</sub> <sub>we</sub> <sub>exa</sub>min<sub>e</sub> <sub>w</sub>h<sub>y</sub> <sub>aes</sub>th<sub>e</sub>ti<sub>c</sub> <sub>co</sub>m<sub>-</sub> petence fails to transfer to culturally situated judgment. In the compact tables, GPT, Gem., Qwen+, Qwen-FT, AQ, and AM abbreviate GPT-5.2, Gemini 3.1 Pro, Qwen3.7-Plus, the CritiqueCanvas-adapted Qwen3-VL, ArtQuant-APDD, <sub>an</sub>d A<sub>r</sub>tiM<sub>use, respec</sub>ti<sub>ve</sub>l<sub>y; ver</sub>ti<sub>ca</sub>l <sub>ru</sub>l<sub>es</sub> f<sub>o</sub>ll<sub>ow</sub> th<sub>e mo</sub>d<sub>e</sub>l <sub>g</sub>r<sub>oups</sub> in S<sub>ec</sub>ti<sub>o</sub>n IV<sub>.</sub>A<sub>.</sub>

V<sub>a</sub>lidit<sub>y</sub> <sub>o</sub>f Criti<sub>que</sub> M<sub>e</sub>tri<sub>cs</sub> W<sub>e</sub> <sub>eva</sub>l<sub>ua</sub>t<sub>e</sub> <sub>w</sub>h<sub>e</sub>th<sub>er</sub> <sub>re</sub>f<sub>erence-</sub>b<sub>ase</sub>d <sub>me</sub>t<sub>r</sub>i<sub>cs re</sub>fl<sub>ec</sub>t <sub>prac</sub>ti<sub>ca</sub>l <sub>cr</sub>iti<sub>que qua</sub>lit<sub>y on</sub> 100 <sub>ran</sub>d<sub>om</sub>l<sub>y</sub> <sub>samp</sub>l<sub>e</sub>d C<sub>ritique</sub>C<sub>anvas</sub> <sub>cases.</sub> 2 h<sub>uman</sub> <sub>eva</sub>l<sub>ua</sub>t<sub>ors ra</sub>t<sub>e</sub> fi<sub>ve represen</sub>t<sub>a</sub>ti<sub>ve mo</sub>d<sub>e</sub>l<sub>s on v</sub>i<sub>sua</sub>l <sub>groun</sub>d<sub>-</sub> i<sub>ng,</sub> <sub>aes</sub>th<sub>e</sub>ti<sub>c</sub> <sub>spec</sub>ifi<sub>c</sub>it<sub>y,</sub> <sub>cr</sub>it<sub>er</sub>i<sub>on</sub> <sub>appropr</sub>i<sub>a</sub>t<sub>eness,</sub> <sub>an</sub>d <sub>ana-</sub> lytical usefulness, with Claude Opus 5 as the MLLM judge <sub>p</sub>rov<sup>idi</sup>n<sub>g</sub> a com<sub>p</sub><sup>l</sup>ementar<sub>y</sub> assessment.

A<sub>s</sub> <sub>s</sub>h<sub>own</sub> i<sub>n</sub> T<sub>a</sub>bl<sub>e</sub> 4<sub>,</sub> di<sub>rec</sub>t <sub>qua</sub>lit<sub>y</sub> <sub>ra</sub>ti<sub>ngs</sub> d<sub>o</sub> <sub>no</sub>t <sub>cons</sub>i<sub>s-</sub> t<sub>en</sub>tl<sub>y</sub> t<sub>rac</sub>k th<sub>e re</sub>f<sub>erence-s</sub>i<sub>m</sub>il<sub>ar</sub>it<sub>y me</sub>t<sub>r</sub>i<sub>cs</sub> i<sub>n</sub> T<sub>a</sub>bl<sub>e</sub> 3<sub>.</sub> Si<sub>nce</sub> <sub>va</sub>lid <sub>cr</sub>iti<sub>ques may emp</sub>h<sub>as</sub>i<sub>ze</sub> dif<sub>eren</sub>t <sub>ev</sub>id<sub>ence an</sub>d i<sub>n</sub>t<sub>er-</sub> <sub>pre</sub>t<sub>a</sub>ti<sub>ons,</sub> l<sub>ex</sub>i<sub>ca</sub>l <sub>an</sub>d <sub>seman</sub>ti<sub>c</sub> <sub>over</sub>l<sub>ap</sub> <sub>cap</sub>t<sub>ure</sub> <sub>on</sub>l<sub>y</sub> <sub>par</sub>t <sub>o</sub>f <sub>cr</sub>iti<sub>que qua</sub>lit<sub>y an</sub>d <sub>s</sub>h<sub>ou</sub>ld b<sub>e comp</sub>l<sub>emen</sub>t<sub>e</sub>d b<sub>y</sub> di<sub>rec</sub>t judgments of grounding, specificity, and usefulness.

<table><tr><td>Rating</td><td>GPT</td><td>Gem.</td><td> $\mathbf { Q } \mathbf { w e n 3 - V L } _ { \mathrm { F T } }$ </td><td>AM</td><td>AQ</td></tr><tr><td>Human</td><td>3.9</td><td>4.1</td><td>2.5</td><td>2.6</td><td>2.3</td></tr><tr><td>MLLM</td><td>5.0</td><td>4.5</td><td>2.5</td><td>2.6</td><td>2.2</td></tr></table>

T<sub>a</sub>bl<sub>e</sub> 4<sub>:</sub> H<sub>uman</sub> <sub>an</sub>d MLLM <sub>ra</sub>ti<sub>ngs</sub> <sub>o</sub>f <sub>overa</sub>ll <sub>cr</sub>iti<sub>que</sub> <sub>qua</sub>l<sub>-</sub> <sup>i</sup>t<sub>y</sub>.
<table><tr><td>Metric</td><td>GPT</td><td>Gem.</td><td>Qwen+</td><td>AQ</td><td>AM</td></tr><tr><td>Correct ↑</td><td>13</td><td>10</td><td>11</td><td>3</td><td>6</td></tr><tr><td>Reverse ↓</td><td>2</td><td>1</td><td>1</td><td>2</td><td>0</td></tr><tr><td>NCU↑</td><td>30.56</td><td>25.00</td><td>27.78</td><td>2.78</td><td>16.67</td></tr></table>

T<sub>a</sub>bl<sub>e</sub> 5<sub>:</sub> R<sub>esu</sub>lt<sub>s o</sub>f th<sub>e</sub> 36<sub>-pa</sub>i<sub>r coun</sub>t<sub>er</sub>f<sub>ac</sub>t<sub>ua</sub>l <sub>au</sub>dit<sub>.</sub>

C<sub>u</sub>lt<sub>u</sub>r<sub>e</sub>-Gr<sub>ou</sub>nd<sub>e</sub>d C<sub>ou</sub>nt<sub>e</sub>rf<sub>ac</sub>t<sub>ua</sub>l S<sub>e</sub>n<sub>s</sub>iti<sub>v</sub>it<sub>y</sub> W<sub>e</sub> f<sub>ur-</sub> th<sub>er exam</sub>i<sub>ne w</sub>h<sub>e</sub>th<sub>er con</sub>t<sub>ex</sub>t<sub>ua</sub>l <sub>aes</sub>th<sub>e</sub>ti<sub>c</sub> d<sub>ec</sub>i<sub>s</sub>i<sub>ons are</sub> <sub>groun</sub>d<sub>e</sub>d i<sub>n cu</sub>lt<sub>ure-</sub>b<sub>ear</sub>i<sub>ng v</sub>i<sub>sua</sub>l <sub>ev</sub>id<sub>ence.</sub> W<sub>e cons</sub>t<sub>ruc</sub>t 36 <sub>pa</sub>i<sub>re</sub>d <sub>coun</sub>t<sub>er</sub>f<sub>ac</sub>t<sub>ua</sub>l <sub>var</sub>i<sub>an</sub>t<sub>s</sub> f<sub>rom</sub> C<sub>on</sub>t<sub>ex</sub>tC<sub>anvas,</sub> k<sub>eep</sub>i<sub>ng</sub> th<sub>e eva</sub>l<sub>ua</sub>ti<sub>on scenar</sub>i<sub>o an</sub>d <sub>overa</sub>ll <sub>v</sub>i<sub>sua</sub>l <sub>presen</sub>t<sub>a</sub>ti<sub>on</sub> fi<sub>xe</sub>d while changing the decisive cue such that the intended judgment changes from No to Yes. These synthetic variants neither <sub>rev</sub>i<sub>se</sub> th<sub>e or</sub>i<sub>g</sub>i<sub>na</sub>l <sub>anno</sub>t<sub>a</sub>ti<sub>ons nor cons</sub>tit<sub>u</sub>t<sub>e an a</sub>dditi<sub>ona</sub>l <sub>accuracy sp</sub>lit<sub>;</sub> th<sub>ey are use</sub>d <sub>on</sub>l<sub>y as a ma</sub>t<sub>c</sub>h<sub>e</sub>d di<sub>agnos</sub>ti<sub>c o</sub>f di<sub>rec</sub>ti<sub>ona</sub>l d<sub>ec</sub>i<sub>s</sub>i<sub>on up</sub>d<sub>a</sub>ti<sub>ng.</sub> W<sub>e repor</sub>t th<sub>e num</sub>b<sub>ers o</sub>f <sub>cor-</sub> rect (No→ Yes) and reverse (Yes→No) updates, as well as Net C<sub>o</sub>rr<sub>ec</sub>t U<sub>p</sub>d<sub>a</sub>t<sub>e co</sub>m<sub>pu</sub>t<sub>e</sub>d <sub>as</sub> $1 0 0 \times \mathrm { ( } N _ { \mathrm { c o r r e c t } } ^ { \mathrm { - } } - N _ { \mathrm { r e v e r s e } } \mathrm { ) / 3 6 }$

A<sub>s s</sub>h<sub>own</sub> i<sub>n</sub> T<sub>a</sub>bl<sub>e</sub> 5<sub>, a</sub>ll th<sub>ree genera</sub>l<sub>-purpose</sub> MLLM<sub>s</sub> <sub>s</sub>h<sub>ow pos</sub>iti<sub>ve ne</sub>t <sub>up</sub>d<sub>a</sub>ti<sub>ng, w</sub>ith NCU <sub>va</sub>l<sub>ues concen</sub>t<sub>ra</sub>t<sub>e</sub>d b<sub>e</sub>t<sub>ween</sub> 25<sub>.</sub>00 <sub>an</sub>d 30<sub>.</sub>56 d<sub>esp</sub>it<sub>e</sub> th<sub>e</sub>i<sub>r su</sub>b<sub>s</sub>t<sub>an</sub>ti<sub>a</sub>ll<sub>y</sub> dif<sub>eren</sub>t b<sub>enc</sub>h<sub>mar</sub>k <sub>accurac</sub>i<sub>es.</sub> Thi<sub>s sugges</sub>t<sub>s</sub> th<sub>a</sub>t <sub>a</sub>b<sub>so</sub>l<sub>u</sub>t<sub>e con</sub>t<sub>ex</sub>t<sub>ua</sub>l <sub>compe</sub>t<sub>ence an</sub>d <sub>respons</sub>i<sub>veness</sub> t<sub>o</sub> d<sub>ec</sub>i<sub>s</sub>i<sub>ve v</sub>i<sub>sua</sub>l <sub>ev</sub>id<sub>ence</sub> <sub>are re</sub>l<sub>a</sub>t<sub>e</sub>d b<sub>u</sub>t <sub>separa</sub>bl<sub>e.</sub> M<sub>ore</sub> i<sub>mpor</sub>t<sub>an</sub>tl<sub>y,</sub> b<sub>o</sub>th <sub>eva</sub>l<sub>ua</sub>t<sub>e</sub>d <sub>aes</sub>th<sub>e</sub>ti<sub>c spec</sub>i<sub>a</sub>li<sub>s</sub>t<sub>s s</sub>h<sub>ow wea</sub>k<sub>er ne</sub>t <sub>up</sub>d<sub>a</sub>ti<sub>ng</sub> th<sub>an every</sub> <sub>g</sub>eneral-purpose model, with ArtQuant exhibitin<sub>g</sub> almost no <sub>measura</sub>bl<sub>e response.</sub> Th<sub>e resu</sub>lt<sub>s</sub> i<sub>n</sub>di<sub>ca</sub>t<sub>e</sub> th<sub>a</sub>t <sub>aes</sub>th<sub>e</sub>ti<sub>c spe-</sub> <sub>c</sub>i<sub>a</sub>li<sub>za</sub>ti<sub>on a</sub>l<sub>one</sub> d<sub>oes no</sub>t <sub>es</sub>t<sub>a</sub>bli<sub>s</sub>h th<sub>e v</sub>i<sub>sua</sub>l<sub>–cu</sub>lt<sub>ura</sub>l bi<sub>n</sub>d<sub>-</sub> ing required for reliable contextual aesthetic judgment.

T<sub>rans</sub>f<sub>er</sub> f<sub>rom</sub> A<sub>es</sub>th<sub>e</sub>ti<sub>c</sub> S<sub>pec</sub>i<sub>a</sub>li<sub>za</sub>ti<sub>on</sub> T<sub>o</sub> t<sub>es</sub>t <sub>w</sub>h<sub>e</sub>th<sub>er</sub> <sub>spec</sub>i<sub>a</sub>li<sub>za</sub>ti<sub>on</sub> f<sub>or conven</sub>ti<sub>ona</sub>l <sub>aes</sub>th<sub>e</sub>ti<sub>c</sub> t<sub>as</sub>k<sub>s</sub> t<sub>rans</sub>f<sub>ers</sub> t<sub>o</sub> contextual suitability judgment, we compare each aesthetic <sub>spec</sub>i<sub>a</sub>li<sub>s</sub>t <sub>w</sub>ith it<sub>s correspon</sub>di<sub>ng</sub> b<sub>ase mo</sub>d<sub>e</sub>l <sub>on</sub> th<sub>e same</sub> C<sub>ontext</sub>C<sub>anvas</sub> <sub>cases.</sub>

A<sub>s</sub> <sub>s</sub>h<sub>own</sub> i<sub>n</sub> Fi<sub>gure</sub> 4<sub>,</sub> <sub>spec</sub>i<sub>a</sub>li<sub>za</sub>ti<sub>on</sub> <sub>s</sub>i<sub>gn</sub>ifi<sub>can</sub>tl<sub>y</sub> i<sub>mproves</sub> ArtQuant and AesExpert over their respective base models, but <sub>y</sub>ields no measurable <sub>g</sub>ain for ArtiMuse or Q-SiT. More-<sub>over,</sub> <sub>s</sub>t<sub>rong</sub> C<sub>r</sub>iti<sub>que</sub>C<sub>anvas</sub> <sub>me</sub>t<sub>r</sub>i<sub>c</sub> <sub>a</sub>li<sub>gnmen</sub>t d<sub>oes</sub> <sub>no</sub>t <sub>con-</sub> <sub>s</sub>i<sub>s</sub>t<sub>en</sub>tl<sub>y</sub> t<sub>rans</sub>l<sub>a</sub>t<sub>e</sub> i<sub>n</sub>t<sub>o</sub> C<sub>on</sub>t<sub>ex</sub>tC<sub>anvas</sub> <sub>accuracy.</sub>

D<sub>ec</sub>i<sub>s</sub>i<sub>ve</sub>-C<sub>ue</sub> Gr<sub>ou</sub>ndin<sub>g</sub> Bi<sub>nary</sub> <sub>accuracy</sub> d<sub>oes</sub> <sub>no</sub>t <sub>s</sub>h<sub>ow</sub> <sub>w</sub>h<sub>e</sub>th<sub>er a mo</sub>d<sub>e</sub>l i<sub>s correc</sub>t f<sub>or</sub> th<sub>e r</sub>i<sub>g</sub>ht <sub>v</sub>i<sub>sua</sub>l <sub>reason.</sub> W<sub>e</sub> th<sub>ere</sub>f<sub>ore</sub> <sub>eva</sub>l<sub>ua</sub>t<sub>e</sub> 100 b<sub>a</sub>l<sub>ance</sub>d C<sub>ontext</sub>C<sub>anvas</sub> <sub>cases</sub> using human judgments of answer correctness, decisive-<sub>cue</sub> id<sub>en</sub>tifi<sub>ca</sub>ti<sub>on, an</sub>d <sub>con</sub>t<sub>ex</sub>t<sub>ua</sub>l <sub>exp</sub>l<sub>ana</sub>ti<sub>on.</sub> E<sub>v</sub>id<sub>ence-</sub> Grounded Accurac<sub>y</sub> (EGA) requires all three.

(a) Effect of aesthetic specialization  
![](images/60595bcb05a3653a64946ca3a8507cd01e0ba622e3e8e5f544294860cedb083a.jpg)  
(b) Critique alignment vs. contextual judgment

![](images/1b2af113c83d946f956f30e60ceb2be601feb4311e9bb38bc9a63a43f16ba2d0.jpg)  
Fi<sub>g</sub>ure 4: Cross-task analysis. (a) Base–s<sub>p</sub>ecialist transfer <sub>on</sub> C<sub>on</sub>t<sub>ex</sub>tC<sub>anvas;</sub> \* d<sub>eno</sub>t<sub>es</sub> B<sub>on</sub>f<sub>erron</sub>i<sub>-correc</sub>t<sub>e</sub>d M<sub>c</sub>N<sub>e-</sub> mar $p < . 0 5 .$ . (b) ContextCanvas accurac<sub>y</sub> a<sub>g</sub>ainst the mean <sub>percen</sub>til<sub>e over s</sub>i<sub>x</sub> C<sub>r</sub>iti<sub>que</sub>C<sub>anvas me</sub>t<sub>r</sub>i<sub>cs.</sub>

<table><tr><td>Metric</td><td>Gem.</td><td>GPT</td><td>Qwen+</td><td>AM</td></tr><tr><td>Accuracy ↑</td><td>83</td><td>65</td><td>42</td><td>21</td></tr><tr><td>EGA↑</td><td>79</td><td>55</td><td>31</td><td>3</td></tr><tr><td>Unsupported ↓</td><td>4.82</td><td>15.38</td><td>26.19</td><td>85.71</td></tr></table>

T<sub>a</sub>bl<sub>e</sub> 6<sub>:</sub> D<sub>ec</sub>i<sub>s</sub>i<sub>ve-cue a</sub>tt<sub>r</sub>ib<sub>u</sub>ti<sub>on on</sub> 100 b<sub>a</sub>l<sub>ance</sub>d <sub>cases.</sub>

A<sub>s s</sub>h<sub>own</sub> i<sub>n</sub> T<sub>a</sub>bl<sub>e</sub> 6<sub>, genera</sub>l<sub>-purpose mo</sub>d<sub>e</sub>l<sub>s preserve</sub> <sub>muc</sub>h <sub>o</sub>f th<sub>e</sub>i<sub>r</sub> <sub>accuracy</sub> <sub>un</sub>d<sub>er</sub> th<sub>e</sub> <sub>s</sub>t<sub>r</sub>i<sub>c</sub>t<sub>er</sub> <sub>groun</sub>di<sub>ng</sub> <sub>cr</sub>it<sub>er</sub>i<sub>on,</sub> <sub>w</sub>h<sub>ereas</sub> A<sub>r</sub>tiM<sub>use</sub> d<sub>rops</sub> f<sub>rom</sub> 21% <sub>accuracy</sub> t<sub>o</sub> 3% EGA<sub>.</sub> U<sub>nsuppor</sub>t<sub>e</sub>d i<sub>s</sub> th<sub>e percen</sub>t<sub>age o</sub>f <sub>correc</sub>t <sub>pre</sub>di<sub>c</sub>ti<sub>ons</sub> th<sub>a</sub>t f<sub>a</sub>il th<sub>e</sub> d<sub>ec</sub>i<sub>s</sub>i<sub>ve-cue</sub> <sub>groun</sub>di<sub>ng</sub> <sub>cr</sub>it<sub>er</sub>i<sub>on.</sub> Thi<sub>s</sub> <sub>gap</sub> i<sub>n</sub>di<sub>ca</sub>t<sub>es</sub> th<sub>a</sub>t fl<sub>uen</sub>t <sub>aes</sub>th<sub>e</sub>ti<sub>c ra</sub>ti<sub>ona</sub>l<sub>es may s</sub>till f<sub>a</sub>il t<sub>o</sub> id<sub>en</sub>tif<sub>y</sub> th<sub>e cu</sub>lt<sub>ure-</sub> b<sub>ear</sub>i<sub>ng ev</sub>id<sub>ence</sub> th<sub>a</sub>t d<sub>e</sub>t<sub>erm</sub>i<sub>nes con</sub>t<sub>ex</sub>t<sub>ua</sub>l <sub>su</sub>it<sub>a</sub>bilit<sub>y.</sub>

## V C<sub>o</sub>n<sub>c</sub>l<sub>us</sub>i<sub>o</sub>n

W<sub>e</sub> i<sub>n</sub>t<sub>ro</sub>d<sub>uce</sub>d A<sub>es</sub>C<sub>anvas, compr</sub>i<sub>s</sub>i<sub>ng</sub> C<sub>ritique</sub>C<sub>anvas</sub> <sub>an</sub>d C<sub>ontext</sub>C<sub>anvas,</sub> t<sub>o eva</sub>l<sub>ua</sub>t<sub>e aes</sub>th<sub>e</sub>ti<sub>c cr</sub>iti<sub>que an</sub>d <sub>con-</sub> t<sub>ex</sub>t<sub>ua</sub>l <sub>su</sub>it<sub>a</sub>bilit<sub>y.</sub> A<sub>cross</sub> <sub>c</sub>l<sub>ose</sub>d<sub>-source,</sub> <sub>open-we</sub>i<sub>g</sub>ht<sub>,</sub> <sub>an</sub>d <sub>aes</sub>th<sub>e</sub>ti<sub>c-spec</sub>ifi<sub>c</sub> <sub>mo</sub>d<sub>e</sub>l<sub>s,</sub> <sub>s</sub>t<sub>rong</sub> <sub>cr</sub>iti<sub>que</sub> <sub>per</sub>f<sub>ormance</sub> d<sub>oes</sub> not reliably translate into context-sensitive judgment: aesth<sub>e</sub>ti<sub>c</sub> <sub>spec</sub>i<sub>a</sub>li<sub>s</sub>t<sub>s</sub> <sub>may</sub> <sub>rema</sub>i<sub>n</sub> <sub>compe</sub>titi<sub>ve</sub> <sub>on</sub> <sub>cr</sub>iti<sub>que</sub> <sub>me</sub>t<sub>-</sub> <sub>r</sub>i<sub>cs</sub> <sub>ye</sub>t <sub>ac</sub>hi<sub>eve</sub> l<sub>ow</sub> <sub>accuracy</sub> <sub>an</sub>d <sub>ex</sub>hibit <sub>over-accep</sub>t<sub>ance</sub> <sub>on</sub> C<sub>ontext</sub>C<sub>anvas.</sub> P<sub>a</sub>i<sub>re</sub>d <sub>an</sub>d di<sub>agnos</sub>ti<sub>c ana</sub>l<sub>yses</sub> f<sub>ur</sub>th<sub>er</sub> <sub>revea</sub>l i<sub>ncons</sub>i<sub>s</sub>t<sub>en</sub>t b<sub>ene</sub>fit<sub>s</sub> f<sub>rom aes</sub>th<sub>e</sub>ti<sub>c</sub> t<sub>un</sub>i<sub>ng an</sub>d li<sub>m</sub>it<sub>e</sub>d <sub>use</sub> <sub>o</sub>f d<sub>ec</sub>i<sub>s</sub>i<sub>ve</sub> <sub>con</sub>t<sub>ex</sub>t<sub>ua</sub>l <sub>ev</sub>id<sub>ence.</sub> Th<sub>ese</sub> fi<sub>n</sub>di<sub>ngs</sub> <sub>suppor</sub>t t<sub>rea</sub>ti<sub>ng</sub> <sub>cu</sub>lt<sub>ura</sub>ll<sub>y</sub> <sub>s</sub>it<sub>ua</sub>t<sub>e</sub>d <sub>su</sub>it<sub>a</sub>bilit<sub>y</sub> <sub>as</sub> <sub>a</sub> di<sub>s</sub>ti<sub>nc</sub>t t<sub>ra</sub>i<sub>n</sub>i<sub>ng</sub> and evaluation objective that requires models to integrate <sub>v</sub>i<sub>sua</sub>l <sub>ev</sub>id<sub>ence w</sub>ith <sub>cu</sub>lt<sub>ura</sub>l k<sub>now</sub>l<sub>e</sub>d<sub>ge,</sub> i<sub>n</sub>t<sub>en</sub>d<sub>e</sub>d <sub>use, an</sub>d d<sub>oma</sub>i<sub>n-spec</sub>ifi<sub>c cons</sub>t<sub>ra</sub>i<sub>n</sub>t<sub>s.</sub>

## R<sub>e</sub>f<sub>erences</sub>

A<sub>n,</sub> X<sub>.;</sub> Xi<sub>e,</sub> Y<sub>.;</sub> Y<sub>ang,</sub> K<sub>.;</sub> Zh<sub>ang,</sub> W<sub>.;</sub> Zh<sub>ao,</sub> X<sub>.;</sub> Ch<sub>eng,</sub> Z<sub>.;</sub> W<sub>a</sub>n<sub>g,</sub> Y<sub>.;</sub> X<sub>u,</sub> S<sub>.;</sub> Ch<sub>e</sub>n<sub>,</sub> C<sub>.;</sub> Zh<sub>u,</sub> D<sub>.;</sub> W<sub>u,</sub> C<sub>.;</sub> T<sub>a</sub>n<sub>,</sub> H<sub>.;</sub> Li<sub>,</sub> C<sub>.;</sub> Yan<sub>g</sub>, J.; Yu, J.; Wan<sub>g</sub>, X.; Qin, B.; Wan<sub>g</sub>, Y.; Yan, Z.; Fen<sub>g</sub>, Z<sub>.;</sub> Li<sub>u,</sub> Z<sub>.;</sub> Li<sub>,</sub> B<sub>.;</sub> <sub>a</sub>nd D<sub>e</sub>n<sub>g,</sub> J<sub>.</sub> 2025<sub>.</sub> LL<sub>a</sub>VA<sub>-</sub>On<sub>e</sub>Vi<sub>s</sub>i<sub>o</sub>n<sub>-</sub> 1<sub>.</sub>5<sub>:</sub> F<sub>u</sub>ll<sub>y</sub> O<sub>pen</sub> F<sub>ramewor</sub>k f<sub>or</sub> D<sub>emocra</sub>ti<sub>ze</sub>d M<sub>u</sub>lti<sub>mo</sub>d<sub>a</sub>l T<sub>ra</sub>i<sub>n</sub>i<sub>ng. ar</sub>Xi<sub>v:</sub>2509<sub>.</sub>23661<sub>.</sub>

B<sub>a</sub>i<sub>,</sub> S<sub>.;</sub> C<sub>a</sub>i<sub>,</sub> Y<sub>.;</sub> Ch<sub>e</sub>n<sub>,</sub> R<sub>.;</sub> Ch<sub>e</sub>n<sub>,</sub> K<sub>.;</sub> Ch<sub>e</sub>n<sub>,</sub> X<sub>.;</sub> Ch<sub>e</sub>n<sub>g,</sub> Z<sub>.;</sub> D<sub>e</sub>n<sub>g,</sub> L<sub>.;</sub> Din<sub>g,</sub> W<sub>.;</sub> G<sub>ao,</sub> C<sub>.;</sub> G<sub>e,</sub> C<sub>.;</sub> G<sub>e,</sub> W<sub>.;</sub> G<sub>uo,</sub> Z<sub>.;</sub> H<sub>ua</sub>n<sub>g,</sub> Q.; Huan<sub>g</sub>, J.; Huan<sub>g</sub>, F.; Hui, B.; Jian<sub>g</sub>, S.; Li, Z.; Li, M.; Li<sub>,</sub> M<sub>.;</sub> Li<sub>,</sub> K<sub>.;</sub> Li<sub>n,</sub> Z<sub>.;</sub> Li<sub>n,</sub> J<sub>.;</sub> Li<sub>u,</sub> X<sub>.;</sub> Li<sub>u,</sub> J<sub>.;</sub> Li<sub>u,</sub> C<sub>.;</sub> Li<sub>u,</sub> Y<sub>.;</sub> Li<sub>u,</sub> D<sub>.;</sub> Li<sub>u,</sub> S<sub>.;</sub> L<sub>u,</sub> D<sub>.;</sub> L<sub>uo,</sub> R<sub>.;</sub> L<sub>v,</sub> C<sub>.;</sub> M<sub>en,</sub> R<sub>.;</sub> M<sub>eng,</sub> L<sub>.;</sub> R<sub>en,</sub> X<sub>.;</sub> R<sub>en,</sub> X<sub>.;</sub> S<sub>ong,</sub> S<sub>.;</sub> S<sub>un,</sub> Y<sub>.;</sub> T<sub>ang,</sub> J<sub>.;</sub> T<sub>u,</sub> J<sub>.;</sub> W<sub>an,</sub> J.; Wan<sub>g</sub>, P.; Wan<sub>g</sub>, P.; Wan<sub>g</sub>, Q.; Wan<sub>g</sub>, Y.; Xie, T.; Xu, Y<sub>.;</sub> X<sub>u,</sub> H<sub>.;</sub> X<sub>u,</sub> J<sub>.;</sub> Y<sub>ang,</sub> Z<sub>.;</sub> Y<sub>ang,</sub> M<sub>.;</sub> Y<sub>ang,</sub> J<sub>.;</sub> Y<sub>ang,</sub> A<sub>.;</sub> Y<sub>u,</sub> B<sub>.;</sub> Zh<sub>ang,</sub> F<sub>.;</sub> Zh<sub>ang,</sub> H<sub>.;</sub> Zh<sub>ang,</sub> X<sub>.;</sub> Zh<sub>eng,</sub> B<sub>.;</sub> Zh<sub>ong,</sub> H<sub>.;</sub> Zh<sub>ou,</sub> J<sub>.;</sub> Zh<sub>ou,</sub> F<sub>.;</sub> Zh<sub>ou,</sub> J<sub>.;</sub> Zh<sub>u,</sub> Y<sub>.; a</sub>nd Zh<sub>u,</sub> K<sub>.</sub> 2025<sub>.</sub> Qwen3-VL Technical Report. arXiv:2511.21631.

Banerjee, S.; and Lavie, A. 2005. METEOR: An Automatic M<sub>e</sub>t<sub>r</sub>i<sub>c</sub> f<sub>or</sub> MT E<sub>va</sub>l<sub>ua</sub>ti<sub>on w</sub>ith I<sub>mprove</sub>d C<sub>orre</sub>l<sub>a</sub>ti<sub>on w</sub>ith H<sub>u</sub>m<sub>a</sub>n J<sub>u</sub>d<sub>g</sub>m<sub>e</sub>nt<sub>s.</sub> In G<sub>o</sub>ld<sub>s</sub>t<sub>e</sub>in<sub>,</sub> J<sub>.;</sub> L<sub>av</sub>i<sub>e,</sub> A<sub>.;</sub> Lin<sub>,</sub> C<sub>.-</sub>Y<sub>.; a</sub>nd Voss, C., eds., Proceedings ofthe ACL Workshop on Intrinsic and Extrinsic Evaluation Measuresfor Machine Translation and/or Summarization, 65–72. Ann Arbor, Michigan: Asso-<sub>c</sub>i<sub>a</sub>ti<sub>on</sub> f<sub>or</sub> C<sub>ompu</sub>t<sub>a</sub>ti<sub>ona</sub>l Li<sub>ngu</sub>i<sub>s</sub>ti<sub>cs.</sub>

C<sub>a</sub>i<sub>,</sub> C<sub>.;</sub> Zh<sub>a</sub>n<sub>g,</sub> R<sub>.;</sub> G<sub>ao,</sub> J<sub>.;</sub> W<sub>u,</sub> K<sub>.;</sub> Y<sub>ap,</sub> K<sub>.-</sub>H<sub>.; a</sub>nd W<sub>a</sub>n<sub>g,</sub> Y<sub>.</sub> 2024<sub>.</sub> T<sub>empora</sub>l <sub>sen</sub>t<sub>ence</sub> <sub>groun</sub>di<sub>ng</sub> <sub>w</sub>ith t<sub>empora</sub>ll<sub>y</sub> <sub>g</sub>l<sub>o</sub>b<sub>a</sub>l textual knowledge. In 2024 IEEE International Conference on Multimedia and Expo (ICME), 1–6. IEEE.

C<sub>ao,</sub> S<sub>.;</sub> Li<sub>,</sub> J<sub>.;</sub> Li<sub>,</sub> X<sub>.;</sub> P<sub>u,</sub> Y<sub>.;</sub> Zh<sub>u,</sub> K<sub>.;</sub> G<sub>ao,</sub> Y<sub>.;</sub> L<sub>uo,</sub> S<sub>.;</sub> Xi<sub>n,</sub> Y.; Qin, Q.; Zhou, Y.; Chen, X.; Zhan<sub>g</sub>, W.; Fu, B.; Qiao, Y.; <sub>an</sub>d Li<sub>u,</sub> Y<sub>.</sub> 2025<sub>a.</sub> U<sub>n</sub>iP<sub>ercep</sub>t<sub>:</sub> T<sub>owar</sub>d<sub>s</sub> U<sub>n</sub>ifi<sub>e</sub>d P<sub>ercep</sub>t<sub>ua</sub>l<sub>-</sub> Level Ima<sub>g</sub>e Understandin<sub>g</sub> across Aesthetics, Qualit<sub>y</sub>, Struct<sub>ure, an</sub>d T<sub>ex</sub>t<sub>ure. ar</sub>Xi<sub>v:</sub>2512<sub>.</sub>21675<sub>.</sub>

C<sub>ao,</sub> S<sub>.;</sub> M<sub>a,</sub> N<sub>.;</sub> Li<sub>,</sub> J<sub>.;</sub> Li<sub>,</sub> X<sub>.;</sub> Sh<sub>ao,</sub> L<sub>.;</sub> Zh<sub>u,</sub> K<sub>.;</sub> Zh<sub>ou,</sub> Y.; Pu, Y.; Wu, J.; Wan<sub>g</sub>, J.; Qu, B.; Wan<sub>g</sub>, W.; Qiao, Y.; Y<sub>ao,</sub> D<sub>.;</sub> <sub>an</sub>d Li<sub>u,</sub> Y<sub>.</sub> 2025b<sub>.</sub> A<sub>r</sub>tiM<sub>use:</sub> Fi<sub>ne-</sub>G<sub>ra</sub>i<sub>ne</sub>d I<sub>mage</sub> A<sub>es</sub>th<sub>e</sub>ti<sub>cs</sub> A<sub>ssessmen</sub>t <sub>w</sub>ith J<sub>o</sub>i<sub>n</sub>t S<sub>cor</sub>i<sub>ng an</sub>d E<sub>xper</sub>t<sub>-</sub>L<sub>eve</sub>l U<sub>n</sub>d<sub>ers</sub>t<sub>an</sub>di<sub>ng. ar</sub>Xi<sub>v:</sub>2507<sub>.</sub>14533<sub>.</sub>

G<sub>ao,</sub> J<sub>.;</sub> Y<sub>ap,</sub> K<sub>.-</sub>H<sub>.;</sub> W<sub>u,</sub> K<sub>.;</sub> Ph<sub>an,</sub> D<sub>.</sub> T<sub>.;</sub> G<sub>arg,</sub> K<sub>.; an</sub>d H<sub>an,</sub> B. S. 2024. Contextual human object interaction understanding from pre-trained large language model. In ICASSP 2024- 2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 13436–13440. IEEE.

GLM<sub>-</sub>V T<sub>eam;</sub> H<sub>ong,</sub> W<sub>.;</sub> Y<sub>u,</sub> W<sub>.;</sub> G<sub>u,</sub> X<sub>.;</sub> W<sub>ang,</sub> G<sub>.; e</sub>t <sub>a</sub>l<sub>.</sub> 2026<sub>.</sub> GLM<sub>-</sub>4<sub>.</sub>5V <sub>an</sub>d GLM<sub>-</sub>4<sub>.</sub>1V<sub>-</sub>Thi<sub>n</sub>ki<sub>ng:</sub> T<sub>owar</sub>d<sub>s</sub> V<sub>er-</sub> <sub>sa</sub>til<sub>e</sub> M<sub>u</sub>lti<sub>mo</sub>d<sub>a</sub>l R<sub>eason</sub>i<sub>ng w</sub>ith S<sub>ca</sub>l<sub>a</sub>bl<sub>e</sub> R<sub>e</sub>i<sub>n</sub>f<sub>orcemen</sub>t Learning. arXiv preprint arXiv:2507.01006.

H<sub>esse</sub>l<sub>,</sub> J<sub>.;</sub> H<sub>o</sub>lt<sub>zman,</sub> A<sub>.;</sub> F<sub>or</sub>b<sub>es,</sub> M<sub>.;</sub> L<sub>e</sub> B<sub>ras,</sub> R<sub>.; an</sub>d Ch<sub>o</sub>i<sub>,</sub> Y<sub>.</sub> 2021<sub>.</sub> CLIPS<sub>core:</sub> A R<sub>e</sub>f<sub>erence-</sub>f<sub>ree</sub> E<sub>va</sub>l<sub>ua</sub>ti<sub>on</sub> M<sub>e</sub>t<sub>r</sub>i<sub>c</sub> f<sub>o</sub>r Im<sub>age</sub> C<sub>ap</sub>ti<sub>o</sub>nin<sub>g.</sub> In M<sub>oe</sub>n<sub>s,</sub> M<sub>.-</sub>F<sub>.;</sub> H<sub>ua</sub>n<sub>g,</sub> X<sub>.;</sub> S<sub>pec</sub>i<sub>a,</sub>

L.; and Yih, S. W.-t., eds., Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, 7514<sub>–</sub>7528<sub>.</sub> O<sub>n</sub>li<sub>ne an</sub>d P<sub>un</sub>t<sub>a</sub> C<sub>ana,</sub> D<sub>om</sub>i<sub>n</sub>i<sub>can</sub> R<sub>epu</sub>bli<sub>c:</sub> A<sub>ssoc</sub>i<sub>a</sub>ti<sub>on</sub> f<sub>or</sub> C<sub>ompu</sub>t<sub>a</sub>ti<sub>ona</sub>l Li<sub>ngu</sub>i<sub>s</sub>ti<sub>cs.</sub>

H<sub>ou,</sub> W<sub>.;</sub> Li<sub>u,</sub> W<sub>.;</sub> H<sub>u,</sub> H<sub>.;</sub> S<sub>un,</sub> X<sub>.;</sub> Y<sub>eung-</sub>L<sub>evy,</sub> S<sub>.; an</sub>d F<sub>an,</sub> H<sub>.</sub> 2026<sub>.</sub> S<sub>ee</sub>i<sub>ng</sub> I<sub>s</sub> B<sub>e</sub>li<sub>ev</sub>i<sub>ng</sub>? A B<sub>enc</sub>h<sub>mar</sub>k f<sub>or</sub> M<sub>u</sub>lti<sub>mo</sub>d<sub>a</sub>l L<sub>arge</sub> L<sub>anguage</sub> M<sub>o</sub>d<sub>e</sub>l<sub>s on</sub> Vi<sub>sua</sub>l Ill<sub>us</sub>i<sub>ons an</sub>d A<sub>noma</sub>li<sub>es.</sub> arXiv preprint arXiv:2602.01816.

Huan<sub>g</sub>, Y.; Shen<sub>g</sub>, X.; Yan<sub>g</sub>, Z.; Yuan, Q.; Duan, Z.; Chen, P.; Li<sub>,</sub> L<sub>.;</sub> Li<sub>n,</sub> W<sub>.;</sub> <sub>an</sub>d Shi<sub>,</sub> G<sub>.</sub> 2024<sub>a.</sub> A<sub>es</sub>E<sub>xper</sub>t<sub>:</sub> T<sub>owar</sub>d<sub>s</sub> M<sub>u</sub>lti<sub>-</sub> <sub>mo</sub>d<sub>a</sub>lit<sub>y</sub> F<sub>oun</sub>d<sub>a</sub>ti<sub>on</sub> M<sub>o</sub>d<sub>e</sub>l f<sub>or</sub> I<sub>mage</sub> A<sub>es</sub>th<sub>e</sub>ti<sub>cs</sub> P<sub>ercep</sub>ti<sub>on.</sub> <sub>ar</sub>Xi<sub>v:</sub>2404<sub>.</sub>09624<sub>.</sub>

Huan<sub>g</sub>, Y.; Yuan, Q.; Shen<sub>g</sub>, X.; Yan<sub>g</sub>, Z.; Wu, H.; Chen, P.; Y<sub>ang,</sub> Y<sub>.;</sub> Li<sub>,</sub> L<sub>.; an</sub>d Li<sub>n,</sub> W<sub>.</sub> 2024b<sub>.</sub> A<sub>es</sub>B<sub>enc</sub>h<sub>:</sub> A<sub>n</sub> E<sub>xper</sub>t B<sub>enc</sub>h<sub>mar</sub>k f<sub>or</sub> M<sub>u</sub>lti<sub>mo</sub>d<sub>a</sub>l L<sub>arge</sub> L<sub>anguage</sub> M<sub>o</sub>d<sub>e</sub>l<sub>s on</sub> I<sub>m-</sub> age Aesthetics Perception. arXivpreprint arXiv:2401.08276.

K<sub>o</sub>n<sub>g,</sub> S<sub>.;</sub> Sh<sub>e</sub>n<sub>,</sub> X<sub>.;</sub> Lin<sub>,</sub> Z<sub>.;</sub> M<sub>ec</sub>h<sub>,</sub> R<sub>.;</sub> <sub>a</sub>nd F<sub>ow</sub>lk<sub>es,</sub> C<sub>.</sub> C<sub>.</sub> 2016<sub>.</sub> Ph<sub>o</sub>t<sub>o</sub> A<sub>es</sub>th<sub>e</sub>ti<sub>cs</sub> R<sub>an</sub>ki<sub>ng</sub> N<sub>e</sub>t<sub>wor</sub>k <sub>w</sub>ith Att<sub>r</sub>ib<sub>u</sub>t<sub>es</sub> and Content Adaptation. In Computer Vision – ECCV 2016, volume 9905 of Lecture Notes in Computer Science, 662– 679<sub>.</sub> S<sub>p</sub>rin<sub>g</sub>er<sub>.</sub>

Li<sub>,</sub> R<sub>.;</sub> Li<sub>,</sub> L<sub>.;</sub> R<sub>en,</sub> S<sub>.;</sub> Ti<sub>an,</sub> H<sub>.;</sub> G<sub>u,</sub> S<sub>.;</sub> Li<sub>,</sub> S<sub>.;</sub> Y<sub>ue,</sub> Z<sub>.;</sub> W<sub>ang,</sub> Y<sub>.;</sub> M<sub>a,</sub> W<sub>.;</sub> Y<sub>ang,</sub> Z<sub>.;</sub> M<sub>a,</sub> J<sub>.;</sub> S<sub>u</sub>i<sub>,</sub> Z<sub>.; an</sub>d L<sub>uo,</sub> F<sub>.</sub> 2026<sub>.</sub> Gr<sub>ou</sub>ndin<sub>g</sub>ME<sub>:</sub> E<sub>xpos</sub>in<sub>g</sub> th<sub>e</sub> Vi<sub>sua</sub>l Gr<sub>ou</sub>ndin<sub>g</sub> G<sub>ap</sub> in MLLMs through Multi-Dimensional Evaluation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2412–2422.

Li<sub>a</sub>n<sub>g,</sub> J<sub>.;</sub> W<sub>u,</sub> K<sub>.;</sub> H<sub>u,</sub> X<sub>.; a</sub>nd Li<sub>u,</sub> T<sub>.</sub> 2026<sub>.</sub> Cibi<sub>c:</sub> Pi<sub>xe</sub>l<sub>-</sub> f<sub>ree</sub> f<sub>oun</sub>d<sub>a</sub>ti<sub>on</sub> <sub>mo</sub>d<sub>e</sub>l f<sub>or</sub> <sub>ro</sub>b<sub>us</sub>t <sub>corrup</sub>t<sub>e</sub>d i<sub>mage</sub> bit<sub>s</sub>t<sub>ream</sub> captioning. Pattern Recognition, 114238.

Li<sub>ao,</sub> M<sub>.;</sub> M<sub>a,</sub> T<sub>.;</sub> <sub>a</sub>nd Zh<sub>a</sub>n<sub>g,</sub> X<sub>.</sub> 2026<sub>.</sub> O<sub>pe</sub>n W<sub>o</sub>rld Im<sub>age</sub> Aesthetic Assessment. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Findings, 9791–9801.

Li<sub>n,</sub> C<sub>.-</sub>Y<sub>.</sub> 2004<sub>.</sub> ROUGE<sub>:</sub> A P<sub>ac</sub>k<sub>age</sub> f<sub>or</sub> A<sub>u</sub>t<sub>oma</sub>ti<sub>c</sub> E<sub>va</sub>l<sub>ua-</sub> tion ofSummaries. In Text Summarization Branches Out, 74– 81<sub>.</sub> B<sub>arce</sub>l<sub>ona,</sub> S<sub>pa</sub>i<sub>n:</sub> A<sub>ssoc</sub>i<sub>a</sub>ti<sub>on</sub> f<sub>or</sub> C<sub>ompu</sub>t<sub>a</sub>ti<sub>ona</sub>l Li<sub>n-</sub> <sub>gu</sub>i<sub>s</sub>ti<sub>cs.</sub>

Li<sub>u,</sub> H<sub>.;</sub> H<sub>ua</sub>n<sub>g,</sub> N<sub>.;</sub> Li<sub>u,</sub> C<sub>.;</sub> Y<sub>a</sub>n<sub>,</sub> J<sub>.;</sub> H<sub>ua</sub>n<sub>g,</sub> H<sub>.;</sub> Yin<sub>g,</sub> J<sub>.;</sub> L<sub>ee,</sub> T<sub>.-</sub>Y<sub>.;</sub> W<sub>a</sub>n<sub>,</sub> P<sub>.;</sub> <sub>a</sub>nd Ji<sub>,</sub> X<sub>.</sub> 2026<sub>.</sub> Brid<sub>g</sub>in<sub>g</sub> C<sub>og</sub>niti<sub>ve</sub> G<sub>ap:</sub> Hi<sub>erarc</sub>hi<sub>ca</sub>l D<sub>escr</sub>i<sub>p</sub>ti<sub>on</sub> L<sub>earn</sub>i<sub>ng</sub> f<sub>or</sub> A<sub>r</sub>ti<sub>s</sub>ti<sub>c</sub> I<sub>mage</sub> A<sub>es</sub>th<sub>e</sub>ti<sub>cs</sub> A<sub>ssessmen</sub>t<sub>. ar</sub>Xi<sub>v:</sub>2512<sub>.</sub>23413<sub>.</sub>

Li<sub>u,</sub> H<sub>.;</sub> Li<sub>,</sub> C<sub>.;</sub> Li<sub>,</sub> Y<sub>.; an</sub>d L<sub>ee,</sub> Y<sub>.</sub> J<sub>.</sub> 2024<sub>.</sub> I<sub>mprove</sub>d b<sub>ase-</sub> lines with visual instruction tuning. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 26296–26306.

M<sub>a</sub>i<sub>,</sub> L<sub>.;</sub> Jin<sub>,</sub> H<sub>.;</sub> <sub>a</sub>nd Li<sub>u,</sub> F<sub>.</sub> 2016<sub>.</sub> C<sub>o</sub>m<sub>pos</sub>iti<sub>o</sub>n<sub>-</sub>Pr<sub>ese</sub>r<sub>v</sub>in<sub>g</sub> Deep Photo Aesthetics Assessment. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 497–506.

M<sub>ora</sub>t<sub>e</sub>lli<sub>,</sub> N<sub>.;</sub> D<sub>av</sub>i<sub>s,</sub> C<sub>.;</sub> Rib<sub>e</sub>i<sub>ro,</sub> L<sub>.</sub> F<sub>.</sub> R<sub>.;</sub> B<sub>yrne,</sub> B<sub>.; an</sub>d I<sub>g</sub>l<sub>es</sub>i<sub>as,</sub> G<sub>.</sub> 2026<sub>.</sub> B<sub>enc</sub>h<sub>mar</sub>ki<sub>ng</sub> D<sub>e</sub>fl<sub>ec</sub>ti<sub>on an</sub>d H<sub>a</sub>ll<sub>uc</sub>i<sub>-</sub> nation in Large Vision-Language Models. arXiv preprint arXiv:2604.12033.

M<sub>urray,</sub> N<sub>.;</sub> M<sub>arc</sub>h<sub>eso</sub>tti<sub>,</sub> L<sub>.; an</sub>d P<sub>erronn</sub>i<sub>n,</sub> F<sub>.</sub> 2012<sub>.</sub> AVA<sub>:</sub> A L<sub>arge-</sub>S<sub>ca</sub>l<sub>e</sub> D<sub>a</sub>t<sub>a</sub>b<sub>ase</sub> f<sub>or</sub> A<sub>es</sub>th<sub>e</sub>ti<sub>c</sub> Vi<sub>sua</sub>l A<sub>na</sub>l<sub>ys</sub>i<sub>s.</sub> I<sub>n</sub>

Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition.

P<sub>ap</sub>i<sub>nen</sub>i<sub>,</sub> K<sub>.;</sub> R<sub>ou</sub>k<sub>os,</sub> S<sub>.;</sub> W<sub>ar</sub>d<sub>,</sub> T<sub>.; an</sub>d Zh<sub>u,</sub> W<sub>.-</sub>J<sub>.</sub> 2002<sub>.</sub> Bl<sub>eu: a</sub> M<sub>e</sub>th<sub>o</sub>d f<sub>or</sub> A<sub>u</sub>t<sub>oma</sub>ti<sub>c</sub> E<sub>va</sub>l<sub>ua</sub>ti<sub>on o</sub>f M<sub>ac</sub>hi<sub>ne</sub> T<sub>rans-</sub> lation. In Isabelle, P.; Charniak, E.; and Lin, D., eds., Proceedings of the 40th Annual Meeting of the Association for Computational Linguistics, 311–318. Philadelphia, Pennsyl-<sub>van</sub>i<sub>a,</sub> USA<sub>:</sub> A<sub>ssoc</sub>i<sub>a</sub>ti<sub>on</sub> f<sub>or</sub> C<sub>ompu</sub>t<sub>a</sub>ti<sub>ona</sub>l Li<sub>ngu</sub>i<sub>s</sub>ti<sub>cs.</sub>

Qi, D.; Zhao, H.; Shi, J.; Jenni, S.; Fan, Y.; Dernoncourt, F.; C<sub>o</sub>h<sub>e</sub>n<sub>,</sub> S<sub>.; a</sub>nd Li<sub>,</sub> S<sub>.</sub> 2025<sub>.</sub> Th<sub>e</sub> Ph<sub>o</sub>t<sub>og</sub>r<sub>ap</sub>h<sub>e</sub>r’<sub>s</sub> E<sub>ye:</sub> T<sub>eac</sub>h<sub>-</sub> in<sub>g</sub> M<sub>u</sub>ltim<sub>o</sub>d<sub>a</sub>l L<sub>a</sub>r<sub>ge</sub> L<sub>a</sub>n<sub>guage</sub> M<sub>o</sub>d<sub>e</sub>l<sub>s</sub> t<sub>o</sub> S<sub>ee, a</sub>nd Cri<sub>-</sub> tique Like Photographers. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, 24807<sub>–</sub>24816<sub>.</sub>

R<sub>e</sub>im<sub>e</sub>r<sub>s,</sub> N<sub>.; a</sub>nd G<sub>u</sub>r<sub>evyc</sub>h<sub>,</sub> I<sub>.</sub> 2019<sub>.</sub> S<sub>e</sub>nt<sub>e</sub>n<sub>ce-</sub>BERT<sub>:</sub> S<sub>e</sub>n<sub>-</sub> t<sub>e</sub>n<sub>ce</sub> Emb<sub>e</sub>ddin<sub>gs us</sub>in<sub>g</sub> Si<sub>a</sub>m<sub>ese</sub> BERT<sub>-</sub>N<sub>e</sub>t<sub>wo</sub>rk<sub>s.</sub> In In<sub>u</sub>i<sub>,</sub> K.; Jiang, J.; Ng, V.; and Wan, X., eds., Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), 3982– 3992<sub>.</sub> H<sub>o</sub>n<sub>g</sub> K<sub>o</sub>n<sub>g,</sub> Chin<sub>a:</sub> A<sub>ssoc</sub>i<sub>a</sub>ti<sub>o</sub>n f<sub>o</sub>r C<sub>o</sub>m<sub>pu</sub>t<sub>a</sub>ti<sub>o</sub>n<sub>a</sub>l Li<sub>ngu</sub>i<sub>s</sub>ti<sub>cs.</sub>

S<sub>a</sub>t<sub>ar,</sub> B<sub>.;</sub> M<sub>a,</sub> Z<sub>.;</sub> I<sub>rawan,</sub> P<sub>.</sub> A<sub>.;</sub> M<sub>u</sub>l<sub>yawan,</sub> W<sub>.</sub> A<sub>.;</sub> Ji<sub>ang,</sub> J<sub>.;</sub> Lim<sub>,</sub> E<sub>.-</sub>P<sub>.; a</sub>nd N<sub>go,</sub> C<sub>.-</sub>W<sub>.</sub> 2025<sub>.</sub> S<sub>ee</sub>in<sub>g</sub> C<sub>u</sub>lt<sub>u</sub>r<sub>e:</sub> A B<sub>e</sub>n<sub>c</sub>h<sub>-</sub> mark for Visual Reasoning and Grounding. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, 22227–22243. Suzhou, China: Asso-<sub>c</sub>i<sub>a</sub>ti<sub>on</sub> f<sub>or</sub> C<sub>ompu</sub>t<sub>a</sub>ti<sub>ona</sub>l Li<sub>ngu</sub>i<sub>s</sub>ti<sub>cs.</sub>

Sin<sub>g</sub>h<sub>,</sub> D<sub>.;</sub> N<sub>ag</sub>r<sub>a</sub>ni<sub>,</sub> A<sub>.;</sub> M<sub>a</sub>nik<sub>a</sub>nt<sub>a</sub>n<sub>,</sub> K<sub>.;</sub> Sin<sub>g</sub>h<sub>,</sub> H<sub>.;</sub> T<sub>ewa</sub>ri<sub>,</sub> D<sub>.;</sub> W<sub>eya</sub>nd<sub>,</sub> T<sub>.;</sub> S<sub>c</sub>hmid<sub>,</sub> C<sub>.;</sub> An<sub>ge</sub>l<sub>ova,</sub> A<sub>.; a</sub>nd D<sub>ave,</sub> S<sub>.</sub> 2026<sub>.</sub> CURVE<sub>:</sub> A B<sub>enc</sub>h<sub>mar</sub>k f<sub>or</sub> C<sub>u</sub>lt<sub>ura</sub>l <sub>an</sub>d M<sub>u</sub>ltili<sub>ngua</sub>l Long Video Reasoning. arXiv preprint arXiv:2601.10649.

T<sub>a</sub>l<sub>e</sub>bi<sub>,</sub> H<sub>.;</sub> <sub>an</sub>d Mil<sub>an</sub>f<sub>ar,</sub> P<sub>.</sub> 2018<sub>.</sub> NIMA<sub>:</sub> N<sub>eura</sub>l I<sub>mage</sub> Assessment. IEEE Transactions on Image Processing, 27(8): 3998<sub>–</sub>4011<sub>.</sub>

W<sub>ang,</sub> W<sub>.;</sub> G<sub>ao,</sub> Z<sub>.;</sub> G<sub>u,</sub> L<sub>.;</sub> P<sub>u,</sub> H<sub>.;</sub> C<sub>u</sub>i<sub>,</sub> L<sub>.;</sub> W<sub>e</sub>i<sub>,</sub> X<sub>.;</sub> Li<sub>u,</sub> Z<sub>.;</sub> Jin<sub>g,</sub> L<sub>.;</sub> Y<sub>e,</sub> S<sub>.;</sub> Sh<sub>ao,</sub> J<sub>.;</sub> W<sub>a</sub>n<sub>g,</sub> Z<sub>.;</sub> Ch<sub>e</sub>n<sub>,</sub> Z<sub>.;</sub> Zh<sub>a</sub>n<sub>g,</sub> H<sub>.;</sub> Yan<sub>g</sub>, G.; Wan<sub>g</sub>, H.; Wei, Q.; Yin, J.; Li, W.; Cui, E.; Chen, G<sub>.;</sub> Di<sub>ng,</sub> Z<sub>.;</sub> Ti<sub>an,</sub> C<sub>.;</sub> W<sub>u,</sub> Z<sub>.;</sub> Xi<sub>e,</sub> J<sub>.;</sub> Li<sub>,</sub> Z<sub>.;</sub> Y<sub>ang,</sub> B<sub>.;</sub> D<sub>uan,</sub> Y<sub>.;</sub> W<sub>ang,</sub> X<sub>.;</sub> H<sub>ou,</sub> Z<sub>.;</sub> H<sub>ao,</sub> H<sub>.;</sub> Zh<sub>ang,</sub> T<sub>.;</sub> Li<sub>,</sub> S<sub>.;</sub> Zh<sub>ao,</sub> X<sub>.;</sub> D<sub>uan,</sub> H<sub>.;</sub> D<sub>eng,</sub> N<sub>.;</sub> F<sub>u,</sub> B<sub>.;</sub> H<sub>e,</sub> Y<sub>.;</sub> W<sub>ang,</sub> Y<sub>.;</sub> H<sub>e,</sub> C<sub>.;</sub> Shi<sub>,</sub> B<sub>.;</sub> H<sub>e,</sub> J<sub>.;</sub> Xi<sub>ong,</sub> Y<sub>.;</sub> L<sub>v,</sub> H<sub>.;</sub> W<sub>u,</sub> L<sub>.;</sub> Sh<sub>ao,</sub> W<sub>.;</sub> Zh<sub>ang,</sub> K<sub>.;</sub> D<sub>eng,</sub> H.; Qi, B.; Ge, J.; Guo, Q.; Zhan<sub>g</sub>, W.; Zhan<sub>g</sub>, S.; Cao, M.; Li<sub>n,</sub> J<sub>.;</sub> T<sub>ang,</sub> K<sub>.;</sub> G<sub>ao,</sub> J<sub>.;</sub> H<sub>uang,</sub> H<sub>.;</sub> G<sub>u,</sub> Y<sub>.;</sub> L<sub>yu,</sub> C<sub>.;</sub> T<sub>ang,</sub> H<sub>.;</sub> W<sub>ang,</sub> R<sub>.;</sub> L<sub>v,</sub> H<sub>.;</sub> O<sub>uyang,</sub> W<sub>.;</sub> W<sub>ang,</sub> L<sub>.;</sub> D<sub>ou,</sub> M<sub>.;</sub> Zh<sub>u,</sub> X.; Lu, T.; Lin, D.; Dai, J.; Su, W.; Zhou, B.; Chen, K.; Qiao, Y<sub>.;</sub> W<sub>ang,</sub> W<sub>.;</sub> <sub>an</sub>d L<sub>uo,</sub> G<sub>.</sub> 2025<sub>.</sub> I<sub>n</sub>t<sub>ern</sub>VL3<sub>.</sub>5<sub>:</sub> Ad<sub>vanc</sub>i<sub>ng</sub> O<sub>pen-</sub>S<sub>ource</sub> M<sub>u</sub>lti<sub>mo</sub>d<sub>a</sub>l M<sub>o</sub>d<sub>e</sub>l<sub>s</sub> i<sub>n</sub> V<sub>ersa</sub>tilit<sub>y,</sub> R<sub>eason</sub>i<sub>ng,</sub> <sub>an</sub>d Efi<sub>c</sub>i<sub>ency. ar</sub>Xi<sub>v:</sub>2508<sub>.</sub>18265<sub>.</sub>

W<sub>u,</sub> H<sub>.;</sub> Zh<sub>a</sub>n<sub>g,</sub> Z<sub>.;</sub> Zh<sub>a</sub>n<sub>g,</sub> W<sub>.;</sub> Ch<sub>e</sub>n<sub>,</sub> C<sub>.;</sub> Li<sub>ao,</sub> L<sub>.;</sub> Li<sub>,</sub> C<sub>.;</sub> G<sub>ao,</sub> Y.; Wan<sub>g</sub>, A.; Zhan<sub>g</sub>, E.; Sun, W.; Yan, Q.; Min, X.; Zhai, G.; and Lin, W. 2024. Q-Ali<sub>g</sub>n: Teachin<sub>g</sub> LMMs for Visual Scoring via Discrete Text-Defined Levels. In Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, 54015<sub>–</sub>54029<sub>.</sub> PMLR<sub>.</sub>

Wu, K.; Li, F.; Liu, W.; Liu, Q.; and Yan<sub>g</sub>, Y. 2026a. Cor-<sub>rup</sub>t<sub>e</sub>d bit<sub>s</sub>t<sub>ream</sub> <sub>seman</sub>ti<sub>c</sub> <sub>un</sub>d<sub>ers</sub>t<sub>an</sub>di<sub>ng</sub> b<sub>y</sub> <sub>a</sub>d<sub>ap</sub>ti<sub>ve-mo</sub>d<sub>a</sub>l large language models. Pattern Recognition, 180: 114151.

Wu, Q.; Yan<sub>g</sub>, X.; Zhou, Y.; Fan<sub>g</sub>, C.; Son<sub>g</sub>, B.; Sun, X.; and Ji<sub>,</sub> R<sub>.</sub> 2026b<sub>.</sub> G<sub>roun</sub>d<sub>e</sub>d Ch<sub>a</sub>i<sub>n-o</sub>f<sub>-</sub>Th<sub>oug</sub>ht f<sub>or</sub> M<sub>u</sub>lti<sub>mo</sub>d<sub>a</sub>l Large Language Models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 33577<sub>–</sub>33587<sub>.</sub>

Xi<sub>ong,</sub> T<sub>.;</sub> G<sub>e,</sub> Y<sub>.;</sub> Li<sub>,</sub> M<sub>.;</sub> Zh<sub>ang,</sub> Z<sub>.;</sub> K<sub>u</sub>lk<sub>arn</sub>i<sub>,</sub> P<sub>.;</sub> W<sub>ang,</sub> K<sub>.;</sub> He, Q.; Zhu, Z.; Liu, C.; Chen, R.; Zhen<sub>g</sub>, T.; Chen, Y.; Wan<sub>g</sub>, X<sub>.;</sub> Zh<sub>a</sub>n<sub>g,</sub> R<sub>.;</sub> Ch<sub>e</sub>n<sub>,</sub> W<sub>.; a</sub>nd H<sub>ua</sub>n<sub>g,</sub> H<sub>.</sub> 2026<sub>.</sub> M<sub>u</sub>lti<sub>-</sub>Crit<sub>:</sub> B<sub>enc</sub>h<sub>mar</sub>ki<sub>ng</sub> M<sub>u</sub>lti<sub>mo</sub>d<sub>a</sub>l J<sub>u</sub>d<sub>ges on</sub> Pl<sub>ura</sub>li<sub>s</sub>ti<sub>c</sub> C<sub>r</sub>it<sub>er</sub>i<sub>a-</sub> Following. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, 8641–8652.

Yan<sub>g</sub>, Y.; Xu, L.; Li, L.; Qie, N.; Li, Y.; Zhan<sub>g</sub>, P.; and Guo, Y. 2022<sub>.</sub> P<sub>ersona</sub>li<sub>ze</sub>d I<sub>mage</sub> A<sub>es</sub>th<sub>e</sub>ti<sub>cs</sub> A<sub>ssessmen</sub>t <sub>w</sub>ith Ri<sub>c</sub>h Attributes. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, 19861–19869.

Y<sub>a</sub>n<sub>g,</sub> Z<sub>.;</sub> W<sub>a</sub>n<sub>g,</sub> J<sub>.;</sub> Zh<sub>a</sub>n<sub>g,</sub> Z<sub>.;</sub> Xi<sub>e,</sub> P<sub>.;</sub> Sh<sub>e</sub>n<sub>g,</sub> X<sub>.;</sub> Ch<sub>e</sub>n<sub>,</sub> P<sub>.; an</sub>d Li<sub>,</sub> L<sub>.</sub> 2026<sub>.</sub> Fi<sub>ne-</sub>G<sub>ra</sub>i<sub>ne</sub>d I<sub>mage</sub> A<sub>es</sub>th<sub>e</sub>ti<sub>c</sub> A<sub>ssess-</sub> <sub>men</sub>t<sub>:</sub> L<sub>earn</sub>i<sub>ng</sub> Di<sub>scr</sub>i<sub>m</sub>i<sub>na</sub>ti<sub>ve</sub> S<sub>cores</sub> f<sub>rom</sub> R<sub>e</sub>l<sub>a</sub>ti<sub>ve</sub> R<sub>an</sub>k<sub>s.</sub> In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 145–155.

Ye, Q.; Xu, H.; Ye, J.; Yan, M.; Hu, A.; Liu, H.; Qian, Q.; Zh<sub>ang,</sub> J<sub>.;</sub> <sub>an</sub>d H<sub>uang,</sub> F<sub>.</sub> 2024<sub>.</sub> <sub>mp</sub>l<sub>ug-ow</sub>l2<sub>:</sub> R<sub>evo</sub>l<sub>u</sub>ti<sub>on</sub>i<sub>z</sub>i<sub>ng</sub> <sub>mu</sub>lti<sub>-mo</sub>d<sub>a</sub>l l<sub>arge</sub> l<sub>anguage</sub> <sub>mo</sub>d<sub>e</sub>l <sub>w</sub>ith <sub>mo</sub>d<sub>a</sub>lit<sub>y</sub> <sub>co</sub>ll<sub>a</sub>b<sub>ora-</sub> tion. In Proceedings of the ieee/cvf conference on computer vision and pattern recognition, 13040–13051.

Yi<sub>,</sub> R<sub>.;</sub> Ti<sub>an,</sub> H<sub>.;</sub> G<sub>u,</sub> Z<sub>.;</sub> L<sub>a</sub>i<sub>,</sub> Y<sub>.-</sub>K<sub>.; an</sub>d R<sub>os</sub>i<sub>n,</sub> P<sub>.</sub> L<sub>.</sub> 2023<sub>.</sub> T<sub>o-</sub> <sub>war</sub>d<sub>s</sub> A<sub>r</sub>ti<sub>s</sub>ti<sub>c</sub> I<sub>mage</sub> A<sub>es</sub>th<sub>e</sub>ti<sub>cs</sub> A<sub>ssessmen</sub>t<sub>:</sub> A L<sub>arge-</sub>S<sub>ca</sub>l<sub>e</sub> Dataset and a New Method. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, 22388<sub>–</sub>22397<sub>.</sub>

Yin<sub>,</sub> Y<sub>.;</sub> Kri<sub>s</sub>hn<sub>a</sub>k<sub>u</sub>m<sub>a</sub>r<sub>,</sub> H<sub>.;</sub> L<sub>ee,</sub> C<sub>.</sub> P<sub>.;</sub> Z<sub>e</sub>n<sub>g,</sub> B<sub>.;</sub> Ch<sub>a</sub>i<sub>,</sub> W<sub>.;</sub> T<sub>o</sub>n<sub>g,</sub> S<sub>.;</sub> Ch<sub>e</sub>n<sub>,</sub> W<sub>.;</sub> X<sub>u,</sub> H<sub>.;</sub> F<sub>u,</sub> X<sub>.;</sub> S<sub>a</sub>r<sub>c</sub>h<sub>,</sub> G<sub>.;</sub> K<sub>o</sub>r<sub>o</sub>l<sub>ova,</sub> A<sub>.;</sub> <sub>an</sub>d Li<sub>u,</sub> Z<sub>.</sub> 2026<sub>.</sub> W<sub>or</sub>ldB<sub>enc</sub>h<sub>:</sub> A Ch<sub>a</sub>ll<sub>eng</sub>i<sub>ng an</sub>d Vi<sub>sua</sub>ll<sub>y</sub> Diverse Multimodal Reasoning Benchmark. arXiv preprint arXiv:2606.06538.

Zhang, L.; Yang, J.; Krishnan, S.; Majmudar, J.; Ge, X.; Puri, P<sub>.;</sub> S<sub>a</sub>r<sub>a</sub>f<sub>,</sub> P<sub>.;</sub> Bh<sub>a</sub>r<sub>gava,</sub> S<sub>.;</sub> Pir<sub>av</sub>i<sub>pe</sub>r<sub>u</sub>m<sub>a</sub>l<sub>,</sub> D<sub>.;</sub> Lin<sub>g,</sub> Y<sub>.;</sub> P<sub>a</sub>n<sub>,</sub> C<sub>.;</sub> Y<sub>u,</sub> H<sub>.;</sub> A<sub>grawa</sub>l<sub>,</sub> A<sub>.; an</sub>d T<sub>seng,</sub> B<sub>.-</sub>H<sub>.</sub> 2026<sub>.</sub> F<sub>rom</sub> Wh<sub>ere</sub> Thi<sub>ngs</sub> A<sub>re</sub> t<sub>o</sub> Wh<sub>a</sub>t Th<sub>ey</sub> A<sub>re</sub> F<sub>or:</sub> B<sub>enc</sub>h<sub>mar</sub>ki<sub>ng</sub> S<sub>pa</sub>ti<sub>a</sub>l<sub>-</sub> Functional Intelligence in Multimodal LLMs. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 12052–12063.

Zhan<sub>g</sub>, T.; Kishore, V.; Wu, F.; Weinber<sub>g</sub>er, K. Q.; and A<sub>r</sub>t<sub>z</sub>i<sub>,</sub> Y<sub>.</sub> 2020<sub>.</sub> BERTS<sub>core:</sub> E<sub>va</sub>l<sub>ua</sub>ti<sub>ng</sub> T<sub>ex</sub>t G<sub>enera</sub>ti<sub>on</sub> <sub>w</sub>ith BERT<sub>. a</sub>rXi<sub>v:</sub>1904<sub>.</sub>09675<sub>.</sub>

Zh<sub>ang,</sub> Z<sub>.;</sub> W<sub>u,</sub> H<sub>.;</sub> Ji<sub>a,</sub> Z<sub>.;</sub> Li<sub>n,</sub> W<sub>.;</sub> <sub>an</sub>d Zh<sub>a</sub>i<sub>,</sub> G<sub>.</sub> 2025<sub>.</sub> Teachin<sub>g</sub> LMMs for Ima<sub>g</sub>e Qualit<sub>y</sub> Scorin<sub>g</sub> and Interpretin<sub>g</sub>. <sub>a</sub>rXi<sub>v:</sub>2503<sub>.</sub>09197<sub>.</sub>

Zh<sub>ou,</sub> S<sub>.;</sub> Ji<sub>a,</sub> B<sub>.;</sub> W<sub>u,</sub> K<sub>.;</sub> Sh<sub>en,</sub> Y<sub>.;</sub> Li<sub>,</sub> T<sub>.;</sub> W<sub>u,</sub> Y<sub>.; an</sub>d Li<sub>n,</sub> S<sub>.</sub> 2026<sub>.</sub> R<sub>eac</sub>tB<sub>enc</sub>h<sub>:</sub> A C<sub>ause-</sub>D<sub>r</sub>i<sub>ven</sub> B<sub>enc</sub>h<sub>mar</sub>k f<sub>or</sub> Multimodal Hallucination via Systematic Evaluation. arXiv preprint arXiv:2605.29579.

Zhou, Z.; Wan<sub>g</sub>, Q.; Lin, B.; Su, Y.; Chen, R.; Tao, X.; Zhen<sub>g</sub>,A<sub>.;</sub> Y<sub>uan,</sub> L<sub>.;</sub> W<sub>an,</sub> P<sub>.;</sub> <sub>an</sub>d Zh<sub>ang,</sub> D<sub>.</sub> 2024<sub>.</sub> UNIAA<sub>:</sub> A

U<sub>n</sub>ifi<sub>e</sub>d M<sub>u</sub>lti<sub>-mo</sub>d<sub>a</sub>l I<sub>mage</sub> A<sub>es</sub>th<sub>e</sub>ti<sub>c</sub> A<sub>ssessmen</sub>t B<sub>ase</sub>li<sub>ne</sub> <sub>an</sub>d B<sub>enc</sub>h<sub>mar</sub>k<sub>. ar</sub>Xi<sub>v:</sub>2404<sub>.</sub>09619<sub>.</sub>

Zh<sub>u,</sub> J<sub>.;</sub> W<sub>a</sub>n<sub>g,</sub> W<sub>.;</sub> Ch<sub>e</sub>n<sub>,</sub> Z<sub>.;</sub> Li<sub>u,</sub> Z<sub>.;</sub> Y<sub>e,</sub> S<sub>.;</sub> G<sub>u,</sub> L<sub>.;</sub> Ti<sub>a</sub>n<sub>,</sub> H<sub>.;</sub> D<sub>ua</sub>n<sub>,</sub> Y<sub>.;</sub> S<sub>u,</sub> W<sub>.;</sub> Sh<sub>ao,</sub> J<sub>.;</sub> G<sub>ao,</sub> Z<sub>.;</sub> C<sub>u</sub>i<sub>,</sub> E<sub>.;</sub> W<sub>a</sub>n<sub>g,</sub> X<sub>.;</sub> C<sub>ao,</sub> Y<sub>.;</sub> Li<sub>u,</sub> Y<sub>.;</sub> W<sub>e</sub>i<sub>,</sub> X<sub>.;</sub> Zh<sub>ang,</sub> H<sub>.;</sub> W<sub>ang,</sub> H<sub>.;</sub> X<sub>u,</sub> W<sub>.;</sub> Li<sub>,</sub> H<sub>.;</sub> W<sub>ang,</sub> J<sub>.;</sub> D<sub>eng,</sub> N<sub>.;</sub> Li<sub>,</sub> S<sub>.;</sub> H<sub>e,</sub> Y<sub>.;</sub> Ji<sub>ang,</sub> T<sub>.;</sub> L<sub>uo,</sub> J<sub>.;</sub> W<sub>a</sub>n<sub>g,</sub> Y<sub>.;</sub> H<sub>e,</sub> C<sub>.;</sub> Shi<sub>,</sub> B<sub>.;</sub> Zh<sub>a</sub>n<sub>g,</sub> X<sub>.;</sub> Sh<sub>ao,</sub> W<sub>.;</sub> He, J.; Xion<sub>g</sub>, Y.; Qu, W.; Sun, P.; Jiao, P.; Lv, H.; Wu, L<sub>.;</sub> Zh<sub>ang,</sub> K<sub>.;</sub> D<sub>eng,</sub> H<sub>.;</sub> G<sub>e,</sub> J<sub>.;</sub> Ch<sub>en,</sub> K<sub>.;</sub> W<sub>ang,</sub> L<sub>.;</sub> D<sub>ou,</sub> M.; Lu, L.; Zhu, X.; Lu, T.; Lin, D.; Qiao, Y.; Dai, J.; and W<sub>a</sub>n<sub>g,</sub> W<sub>.</sub> 2025<sub>.</sub> Int<sub>e</sub>rnVL3<sub>:</sub> E<sub>xp</sub>l<sub>o</sub>rin<sub>g</sub> Ad<sub>va</sub>n<sub>ce</sub>d Tr<sub>a</sub>inin<sub>g</sub> <sub>an</sub>d T<sub>es</sub>t<sub>-</sub>Ti<sub>me</sub> R<sub>ec</sub>i<sub>pes</sub> f<sub>or</sub> O<sub>pen-</sub>S<sub>ource</sub> M<sub>u</sub>lti<sub>mo</sub>d<sub>a</sub>l M<sub>o</sub>d<sub>e</sub>l<sub>s.</sub> <sub>ar</sub>Xi<sub>v:</sub>2504<sub>.</sub>10479<sub>.</sub>

## Supplementary Material

# AesCanvas: A Large-Scale Dataset and Benchmark for Aesthetic Critique and Contextual Suitability

## Supplement Contents

S1 Supplement Overview and Positioning 2   
S1.1 Positioning Relative to Prior Datasets 2   
S2 Dataset and Evaluation Materials 3   
S2.1 Dataset Documentation . 3   
S2.2 Prompts 7   
S2.3 Evaluation Rubrics and Metrics 15   
S2.4 Implementation Details 19   
S3 Additional Analyses and Insights 20   
S3.1 Model-Output Analysis . 20   
S3.2 Extended and Robustness Analyses 23

## S1 Supplement Overview and Positioning

This supplement documents AesCanvas and additional diagnostics. Section S2 provides the dataset, prompt, evaluation, and implementation materials; Section S3 presents model-output, robustness, and transfer analyses.

## S1.1 Positioning Relative to Prior Datasets

Table S1 compares published dataset and supervision afordances. A checkmark requires explicit support in the corresponding paper or dataset documentation; a triangle denotes limited or indirect support. Usecontext judgment specifically requires an auditable decision about whether an image is appropriate for a stated purpose, audience, or convention, rather than generic context reasoning or intrinsic aesthetic scoring.

The comparison covers established photographic-aesthetics datasets (Murray, Marchesotti, and Perronnin, 2012; Kong et al., 2016; Yang et al., 2022), AI-generated-image quality datasets (Li et al., 2024; Wang et al., 2023), and recent multimodal aesthetic-critique and assessment resources (Huang et al., 2024b,a; Qi et al., 2025; Cao et al., 2025; Liu et al., 2026).

Table S1: Comparison of media coverage and aesthetic supervision.
<table><tr><td>Dataset</td><td>Scale</td><td>Photo- graphy</td><td>Painting / art</td><td>AIGC / virtual</td><td>Long-form critique</td><td>Structured dimensions</td><td>Grounded explanation</td><td>Use-context judgment</td></tr><tr><td>AVA</td><td>&gt;250K images</td><td>√</td><td>一</td><td>一</td><td></td><td>△</td><td></td><td></td></tr><tr><td>AADB</td><td>10K images</td><td>√</td><td>一</td><td>一</td><td>一</td><td>√</td><td></td><td>一</td></tr><tr><td>PARA</td><td>31,220 images</td><td>√</td><td>一</td><td>一</td><td>1</td><td>√</td><td>一</td><td>一</td></tr><tr><td>AGIQA-3K</td><td>2,982 AGIs</td><td>-</td><td>一</td><td>√</td><td>一</td><td>△</td><td>一</td><td>一</td></tr><tr><td>AIGCIQA- 2023</td><td>2,400 AGIs</td><td>一</td><td>一</td><td>√</td><td>一</td><td>√</td><td>一</td><td></td></tr><tr><td>AesBench/ EAPD</td><td>2,800 images; 11,200 annotations</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td><td></td></tr><tr><td>AesMMIT</td><td>21,904 images; 409K instructions</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td><td></td></tr><tr><td>Photo- Critique</td><td>450K images; 2.4M</td><td>√</td><td>一</td><td>一</td><td>√</td><td>△</td><td>√</td><td></td></tr><tr><td>ArtiMuse- 10K</td><td>samples 10K images</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td><td></td></tr><tr><td>RAD/ ArtQuant</td><td>70K structured descriptions</td><td>一</td><td>√</td><td>一</td><td>√</td><td>√</td><td>√</td><td></td></tr><tr><td>AesCanvas (ours)</td><td>54.3K images; 519K critique pairs; 301 context cases</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td></tr></table>

Note. –denotes that the afordance is not established in the published dataset design. AVA provides score distributions and semantic/style labels; AGIQA-3K provides perceptual-quality and text–image-alignment scores; and PhotoCritique provides broad photographic feedback rather than a fixed multi-dimensional annotation schema. “Painting / art” includes broader artistic imagery, while “AIGC / virtual” includes generated and other virtual imagery.

Prior resources already provide substantial supervision for intrinsic aesthetic perception and critique. AesCanvas complements them by coupling long-form critique with a closed-form benchmark in which cultural, narrative, functional, or domain-specific evidence can change whether an image is suitable for a concrete use.

## S2 Dataset and Evaluation Materials

This section documents the record schemas, prompt templates and representative intervention specifications, evaluation procedures, and implementation details needed to interpret AesCanvas and the reported measurements.

## S2.1 Dataset Documentation

## S2.1.1 AesCanvas Overview

AesCanvas contains two complementary components derived from a shared image pool. Table S2 summarizes their data units, scales, outputs, and evaluation roles.

Table S2: Suite-level data card for the two AesCanvas components.
<table><tr><td>Field</td><td>CritiqueCanvas</td><td>ContextCanvas</td></tr><tr><td>Primary task</td><td>tique</td><td>Long-form, multi-dimensional aesthetic cri- Contextual aesthetic suitability judgment</td></tr><tr><td>Data unit</td><td>response</td><td>Image, instruction, selected dimensions, and Image(s), use scenario, options, gold label, ra- tionale, and provenance</td></tr><tr><td>Scale</td><td>519,136 instruction-response pairs</td><td>301 cases using 302 unique images</td></tr><tr><td>Output</td><td>Open-ended critique</td><td>291 two-option and 10 three-option decisions</td></tr><tr><td>Coverage</td><td>Photography, painting, and virtual imagery</td><td>Painting, sculpture, illustration/comics, anima- tion/digital imagery, photography, and film</td></tr><tr><td>Evaluation role</td><td>6,000-case generation evaluation</td><td>Fixed-order diagnostic benchmark</td></tr></table>

The shared pool comprises 21,294 photographs, 17,939 paintings, and 15,067 virtual images. Photography includes natural scenes, architecture, portraiture, commercial imagery, and other camera-based work; painting contains primarily Western classical and related painted work; and virtual imagery includes game scenes, animation-style illustration, and AI-generated content. Images were collected from publicly accessible web sources, museums, and cultural-heritage collections. Low-quality, semantically uninformative, non-compliant, duplicate, and near-duplicate candidates were removed. CritiqueCanvas conversations derived from the same image are grouped before splitting so that related records do not cross its train/evaluation partitions. ContextCanvas is used as a fixed diagnostic set rather than a training split.

For documentation clarity, we provide the dataset cards, record schemas, prompts, metric definitions, evaluation and parsing procedures, and representative examples for both components. Task-specific model inputs, supervision fields, rationales, and provenance metadata are described below.

## S2.1.2 CritiqueCanvas

CritiqueCanvas uses ten shared dimensions: Content/Narrative, Composition, Color, Lighting, Lines/Brushstrokes, Style, Emotion, Technique, Symbolism, and Visual Appeal. Only image-relevant dimensions are selected for a given record. The shared dimensions are complemented by domain-conditioned criteria: photography emphasizes exposure, perspective, depth of field, focus, motion, and compositional control; painting emphasizes line, brushwork, genre, style, symbolism, and emotion; and virtual imagery emphasizes character design, scene construction, rendering consistency, narrative setting, and stylization.

The CritiqueCanvas record schema identifies the image and domain, selected dimensions, instruction, target response, conversation group, split, and associated provenance/governance metadata. The evaluation loader consumes a compact conversation form with id, image, and conversations; it takes the first user/human turn as the instruction and the first assistant/GPT turn as the reference. The image and instruction form the model input, while the reference is withheld until scoring. Each row records sample id, relative and resolved image paths, the exact rendered prompt, reference, prediction, model, task, and timestamp. Each six-pair construction dialogue was converted into six pair-level records sharing a conversation-group identifier; the evaluation export therefore contains one human–assistant pair per record. Source, license, selected-dimension, split, audit, and governance fields are retained as metadata but are never supplied as model input. Representative examples spanning photography, illustration, and painting, as well as both open-ended and dimension-focused instruction functions, are shown in Fig. S1.

Gemini performs dimension-aware planning, GLM generates the corresponding domain-conditioned critique annotations, and an independent Claude Opus 5 verification stage screens instruction consistency, visual grounding, and response quality. Five human reviewers then independently inspected a stratified sample of 1,000 retained pairs using the rubric detailed below. The audit yielded 95.3% acceptance and 97.0% raw agreement.

## S2.1.3 ContextCanvas

ContextCanvas was sampled from the same 54,300-image pool. The 1,060 initial candidate cases were collected primarily through a call circulated within the university. Graduate research assistants, doctoral researchers, and faculty spanning computer science and electrical engineering, visual arts and art history, design and visual communication, and cultural and media studies contributed source ideas and image–use proposals. A separate multidisciplinary panel of four PhD-level researchers developed these contributions into formal candidate cases and reviewed them through iterative refinement, standardization, cross-disciplinary checking, and final adjudication. A candidate was retained only if it satisfied aesthetic centrality, context dependence, visual necessity, decisive evidence, scenario naturalness, answerability, and anti-shortcut validity. Qwen3.7-Plus was used only as a development-time probe for trivial, ambiguous, or shortcut-prone candidates; model failure was never an inclusion condition. The four-person review panel finalized all questions, options, gold labels, source-grounded rationales, and source records.

The 301 retained cases comprise 300 single-image cases and one paired-image case, covering 302 unique images. Each case record stores the stable evaluation ID, original case ID, image descriptor(s), question, options, gold option ID and text, source-grounded rationale, source basis, and provenance/rights fields. Only the image(s), question, and fixed options are model input; labels, rationales, source titles, URLs, development metadata, and model-probe results are excluded.

A label-balanced demonstration of retained cases is shown in Fig. S2. The examples are organized by the mechanism that makes the aesthetic-use decision non-trivial: cultural or historical alignment; narrative meaning that reverses attractive, serene, or balanced surface qualities; apparent visual mismatch that strengthens a communicative purpose; and medium- or story-specific framing. Every example distinguishes the visible cue, the contextual knowledge it activates, and the resulting visual-use decision. This presentation makes clear that the task is neither artwork-title recall nor context-free cultural question answering.

Expert review also removed coherent human-drafted questions that failed one of the retention requirements. Figure S3 shows three such boundaries: a bridal image whose conspicuous negative afect makes the answer available through coarse emotion recognition; The Wolfand the Crane, whose decisive betrayal occurs outside the depicted moment and therefore weakens visual necessity; and the historical “DON’T MIX ’EM” poster, for which a contemporary campaign can defensibly reject the sensational design, preventing a uniquely compelled gold label. These examples demonstrate that professional wording and a plausible intended answer were insuficient for retention. They are illustrative and are not used to estimate a distribution of rejection reasons.

![](images/2283911417d21fd4bd82835ba56d18b1a79e6d95e4352f8d7e3b56e848fc0356.jpg)  
Figure S1: Representative CritiqueCanvas examples across photography, illustration, and painting. Each example pairs an image with its rendered query and the corresponding aesthetic focus. The examples include both broad image-appreciation prompts and prompts targeting specific dimensions such as light and shadow, atmosphere, tonal treatment, and formal analysis

## S2.1.4 Provenance and Governance

For each ContextCanvas image, provenance metadata include the known title, creator, provider, source URL, license note, and redistribution status. Missing creator names are preserved as unknown rather than inferred. Synthetic counterfactual edits are clearly labeled as research diagnostics and are not treated as natural benchmark images or training annotations.

The datasets necessarily reflect the source pool and the expertise of the case contributors and review panel. ContextCanvas is an expert diagnostic benchmark rather than exhaustive coverage of cultures, audiences, or design practice, and a source-grounded rationale does not eliminate every historically or culturally contestable interpretation. CritiqueCanvas similarly permits multiple valid analyses of the same image; its reference response is supervision, not the only legitimate critique.

![](images/c66588c9ea1cd1fc30c8e390c9cd6c75f6ecb6cf301524410b86c5fbe5c11bc9.jpg)  
Figure S2: Representative retained ContextCanvas cases. Top row, from left to right: HB208 (Mid-Autumn cultural fit), HB014 (historical military identity), and HB159 (restorative return versus the Lotus-Eaters’ withdrawal). Bottom row: HB155 (fiction–reality confusion), HB181 (teamwork under visual tension), and HB186 (addiction in a refined setting). Each card separates visible form, use scenario, contextual evidence, decision, and gold label.

![](images/b194be985279f8fbe54cce6e82c87f6577399fefb3230e3e1dab218c7d1b820e.jpg)

Question: A bridal salon is considering this image for “A Joyful Yes, Freely Chosen.” Do you agree that the image is suitable without qualification for this campaign? Intended gold: No

![](images/84bc916ba8fdf469d56a0caffc87d6bda30e378f28a02779043f3c5f2c5ba029.jpg)

Question: A medical association is   
considering this image for “Trust Makes   
Difficult Care Possible.” Do you agree that   
the image is suitable without qualification   
for this campaign?   
Intended gold: No   
Question: A city road-safety programme is   
choosing the lead image for a campaign   
about the danger of combining alcohol and   
driving. Should it be considered an ideal   
promotional image?   
Intended gold: Yes

![](images/37407d0413693c98c3b28d00b2c8da96726665b1232187d7692318b1130a869d.jpg)

![](images/5f1f1364c537383b2dfd87c6465f37657a5f712e3e77032264e5e6f247f409bd.jpg)

Reasoning: Insufficiently challenging for the benchmark: conspicuous negative affect allows correct answers via coarse emotion recognition, bypassing cultural context and fine-grained aesthetic evidence.

![](images/0466ebce12cac11bce831659e55e596e8387758938414d45ee8317cae8d8cf2d.jpg)

Reasoning: Betrayal occurs after the depicted moment, unseen in the image. Item relies on recalling the full story, risking literary retrieval over image-grounded judgment.

![](images/5cc4a38d51b795a9d9e6bd0c91406baf893496b6ee778685c1c75ad735394ead.jpg)

Reasoning: A contemporary city campaign may reject old sensational design despite understanding its purpose. The consultant's objection is strategically defensible, so "ideal" has no single compelled answer.

Figure S3: Human-drafted candidates excluded after expert review. From left to right: R41Q05 was insuficiently challenging because conspicuous negative afect enabled a coarse shortcut; R41Q08 weakened visual necessity because the decisive betrayal occurs outside the depicted moment; and R69Q06 lacked a uniquely compelled gold label for a contemporary campaign.

## S2.2 Prompts

We report the prompts used in dataset construction and in the experiments described in the main paper.

## S2.2.1 CritiqueCanvas Prompts

## CritiqueCanvas <sup>·</sup> Planning Prompt

System Message

You are a Planner MLLM for long-form aesthetic critique construction. Your role is to inspect the image and select the most relevant evaluation dimensions before another model writes the final critique. Do not write the final answer.

Use the following ten shared aesthetic dimensions for all images:

1. Content and Narrative: subject matter, narrative, visual theme, contextual meaning, and what the image is primarily about.

2. Composition: spatial organization, focal point, balance, perspective, framing, visual hierarchy, leading lines, and arrangement of elements.

3. Color: palette, harmony, contrast, saturation, brightness, color balance, and the emotional or symbolic function of color.

4. Lighting: light source, illumination, highlights, shadows, contrast, chiaroscuro, tonal relations, and the way light shapes volume or mood.

5. Lines and Brushstrokes: contour, line quality, mark direction, stroke layering, surface handling, and their contribution to form, texture, rhythm, or expression.

6. Style: medium category, visual language, genre, historical or cultural reference, personal style, and suitability of style to the image purpose.

7. Emotion: mood, emotional tone, expressive force, atmosphere, tension, serenity, intimacy, or dramatic effect.

8. Technique: technical execution, rendering skill, perspective control, texture handling, detail, focus, depth of field, digital or photographic technique, and other craft choices.

9. Symbolism: iconography, metaphor, culturally situated signs, repeated motifs, and the conceptual meaning carried by visible elements.

10. Visual Appeal: immediate visual impact, harmony, memorability, beauty, refinement, visual richness, and overall aesthetic attraction.

Apply image-type-specific supplements when relevant:

Painting images: additionally consider lines and brushstrokes, painterly surface, symbolic or iconographic elements, painting genres, art-historical references, and other painting-specific themes.

\- Virtual or illustration images: additionally consider characters, character interaction, story world, implied narrative, fantasy or symbolic setting, and the symbolism behind the image.

Photographic images: additionally consider photographic framing, lens   
perspective, focus, depth of field, exposure, timing, and post-processing   
when they are visually relevant.   
Select the five most relevant dimensions. You may select fewer than five if   
appropriate. Prefer dimensions supported by clear visual evidence in the image.   
Return only the selected dimensions and a brief visual reason for each. Do   
not write the final critique. Do not use markdown tables.   
User Message   
No separate textual user-message template is defined. The image is supplied as the multimodal input.   
Required Output   
Selected Dimensions:   
Dimension: ¡dimension name¿   
Reason: ¡brief reason based on visible evidence¿   
===   
Dimension: ¡dimension name¿   
Reason: ¡brief reason based on visible evidence¿   
[repeat for every selected dimension]

## CritiqueCanvas <sup>·</sup> Photography Annotation Prompt

System Message   
You are an Annotator MLLM and professional photography critic. You will receive   
a photograph and the dimensions selected by a Planner MLLM. Write the final   
aesthetic annotation according to those selected dimensions.   
The shared dimensions are content and narrative, composition, color, lighting,   
lines and brushstrokes, style, emotion, technique, symbolism, and visual   
appeal. For photographs, pay special attention to exposure, perspective,   
depth of field, focus, motion, timing, framing, compositional control, light   
quality, and post-processing when these features are visually relevant.   
Generate a dialogue between a questioner and an expert critic. Ask six   
different questions and give corresponding answers. The questions should be   
guided by the selected dimensions, and one question-answer pair should discuss   
how the photograph may be improved. Answers should cite concrete visual   
evidence from the image and avoid generic comments.

Just give the question and answer, no other text, and do not use markdown   
format.   
User Message   
The current image is attached, and the planner response is inserted verbatim as {planner output}.   
Required Output   
Question:   
¡question 1¿   
===   
Answer:   
¡answer 1¿   
===   
Question:   
¡question 2¿   
===   
Answer:   
¡answer 2¿   
===   
. [continue in the same format for six question--answer pairs in total]

CritiqueCanvas <sup>·</sup> Illustration Annotation Prompt

System Message

You are an Annotator MLLM and professional illustration critic. You will receive a virtual or illustration image and the dimensions selected by a Planner MLLM. Write the final aesthetic annotation according to those selected dimensions.

The shared dimensions are content and narrative, composition, color, lighting, lines and brushstrokes, style, emotion, technique, symbolism, and visual appeal. For virtual or illustration images, pay special attention to character design, character interaction, scene construction, rendering consistency, story world, implied narrative, symbolic or fantasy setting, and stylization when these features are visually relevant.

Generate a dialogue between a questioner and an expert critic. Ask six different questions and give corresponding answers. The questions should be guided by the selected dimensions, and one question-answer pair should discuss how the illustration may be improved. Answers should cite concrete visual evidence from the image and avoid generic comments.

Just give the question and answer, no other text, and do not use markdown   
format.   
User Message   
The current image is attached, and the planner response is inserted verbatim as {planner output}.   
Required Output   
Question:   
¡question 1¿   
===   
Answer:   
¡answer 1¿   
===   
Question:   
¡question 2¿   
===   
Answer:   
¡answer 2¿   
= ===   
[continue in the same format for six question--answer pairs in total]

## CritiqueCanvas <sup>·</sup> Painting Annotation Prompt

## System Message

You are an Annotator MLLM and professional painting critic. You will receive a painting image and the dimensions selected by a Planner MLLM. Write the final aesthetic annotation according to those selected dimensions.

The shared dimensions are content and narrative, composition, color, lighting, lines and brushstrokes, style, emotion, technique, symbolism, and visual appeal. For painting images, pay special attention to line, brushwork, painterly surface, impasto or smooth blending, genre, style, symbolic or iconographic elements, art-historical references, and emotional expression when these features are visually relevant.

Generate a dialogue between a questioner and an expert critic. Ask six different questions and give corresponding answers. The questions should be guided by the selected dimensions, and one question-answer pair should discuss how the painting may be improved. Answers should cite concrete visual evidence from the image and avoid generic comments.

Just give the question and answer, no other text, and do not use markdown   
format.   
User Message   
The current image is attached, and the planner response is inserted verbatim as {planner output}.   
Required Output   
Question:   
¡question 1¿   
===   
Answer:   
¡answer 1¿   
===   
Question:   
¡question 2¿   
===   
Answer:   
¡answer 2¿   
===   
[continue in the same format for six question--answer pairs in total]

## CritiqueCanvas Verification Prompt

System Message   
You are verifying an image-grounded aesthetic-critique record. Inspect the   
supplied image, instruction, and candidate response.   
Evaluate:   
1. Instruction consistency: the instruction is appropriate and answerable   
from the image, and the response follows the requested task and format.   
2. Visual grounding: substantive claims are supported by visible evidence,   
without materially invented objects, actions, colors, spatial relations,   
techniques, or events.   
3. Response quality: the response provides coherent, useful, and   
image-specific aesthetic analysis rather than generic or repetitive prose.   
A blocking issue includes an instruction-image mismatch, failure to perform   
the requested task, material visual hallucination, a response so generic or   
incomplete that it does not fulfill the instruction, or an explicit safety or   
compliance problem observable in the supplied record.

Accept if and only if all three criteria pass and no blocking issue is present.   
If any criterion fails, the decision must be reject and the issues list must   
contain at least one corresponding blocking issue. If no issue is identified,   
return an empty issues list.   
Return JSON only. Do not use external tools or retrieve outside information.   
Judge the record from the supplied image, instruction, and response.   
User Message   
Instruction:   
–instruction   
Candidate response:   
–response   
Required Output   
”instructionconsistency”: ”pass—fail”,   
”visualgrounding”: ”pass—fail”,   
”responsequality”: ”pass—fail”,   
”issues”: [   
”criterion”: ”instructionconsistency—visualgrounding—   
responsequality—compliance”,   
”severity”: ”blocking—nonblocking”,   
”description”: ”¡brief evidence-based description¿”   
],   
”decision”: ”accept—reject”,   
”reason”: ”¡one concise sentence¿”

## S2.2.2 ContextCanvas Prompts

## ContextCanvas <sup>·</sup> Evaluation Prompt

System Message   
You are being evaluated on contextual aesthetic judgment. Answer only from the   
supplied visual content and prompt. Do not use web browsing, reverse-image   
search, image-recognition tools, metadata inspection, external retrieval, or

other tools. Evaluate whether the visual is suitable for the stated real-world   
aesthetic context, including relevant cultural and narrative meaning.   
User Message   
Do not call or simulate any external tool. Inspect the supplied image(s), then   
make the requested contextual aesthetic judgment.   
Question:   
–question   
Options:   
–optionid1. –optiontext1   
–optionid2. –optiontext2   
Return the single best option ID, a concise reason, and confidence.   
Required Output   
”answer”: ”¡one supplied option ID¿”,   
”reason”: ”¡concise reason¿”,   
”confidence”: ”low—medium—high”

Where supported, provider-side structured-output enforcement was requested for the required JSON schema. The reason and confidence fields are retained for qualitative analysis but do not afect exactmatch grading, which is based solely on the predicted option ID.

Counterfactual editing prompts. Each of the 36 edits replaces the decisive incompatible cue with a scenario-compatible alternative while aiming to preserve the source medium, composition, palette, viewpoint, and non-target content as closely as possible. We use GPT Image 2 for all edits. Pair-specific interventions were specified individually. For example, CF01 replaces Japanese-inspired wafuku and tea props with Chinese hanfu, a Chinese tea set, mooncakes, and a palace lantern; CF18 replaces the governess’s distressing letter with an open lesson book and makes the children engage with her while retaining the painted nursery and period style. All edited images were manually checked for successful cue replacement and preservation of non-target content; invalid edits were regenerated.

Modality diagnostic. Motivated by caption-utility evaluation that treats captions as image surrogates (Yang et al., 2026), we additionally evaluate text-only, identity- and verdict-free neutral-caption, and decisive-visible-cue conditions using the same questions, options, output schema, and scoring protocol. The condition construction and aggregate results are reported below.

## S2.2.3 Evaluation and Judge Prompts

Human critique-rating instructions. Two human evaluators independently assessed five anonymous critiques for each image without access to model identities or to each other’s ratings. For each critique, they assigned a binary judgment on four criteria: visual grounding, aesthetic specificity, criterion appropriateness, and analytical usefulness. A criterion received one point when the critique was comparatively strong on that dimension within the same image–prompt batch, and zero points when it was comparatively weak. The final human score was one plus the number of satisfied criteria, yielding a score from 1 to 5. Thus, a zero on an individual criterion denotes relative weakness among responses to the same prompt rather than the complete absence of that quality.

MLLM critique judge. Following rubric-based model evaluation (Liu et al., 2023; Chen et al., 2024), Claude Opus 5 provides a complementary assessment. The judge assigns four criterion-specific scores and one holistic overall-quality score, each from 1 to 5. For each case, the five candidate critiques are anonymized and randomly permuted before evaluation, then assigned temporary aliases A–E; the alias-to-model mapping is restored after scoring. The reference critique is withheld. The fifth, holistic score is reported as the MLLM score in Table 4 of the main paper.

Direct Critique-Quality Judge Prompt   
System Message   
You are an expert visual-aesthetic critique evaluator. Rate model-written   
critiques for practical aesthetic analysis quality. You must judge from the   
image and the question only. Do not reward similarity to any hidden reference   
answer. Do not reward length by itself. Prefer critiques that point to concrete   
visual evidence in the image, use media-appropriate aesthetic criteria, and   
produce useful interpretation.   
User Message   
Evaluate five anonymous critiques of the same image.   
Question:   
–prompt   
Rubric, assign five scores from 1 to 5:   
- visualgrounding: Are claims supported by concrete visual evidence?   
- aestheticspecificity: Is the critique specific rather than templated?   
- criterionappropriateness: Are the criteria appropriate for the medium,   
image, and question?   
- analyticalusefulness: Does the critique provide explanatory or actionable   
aesthetic insight?   
- overallquality: Considering the four criteria together, how strong is the   
critique as a whole?

Do not reward length or wording similarity to a hidden reference. Penalize   
generic quality templates. Use the full 1--5 range.   
Return the scores in this order: visual grounding, aesthetic specificity,   
criterion appropriateness, analytical usefulness, and overall quality.   
Anonymous critiques:   
–critiqueAtoE   
Required Output   
–”sampleid”:”...”,”scores”:–”A”:[1,1,1,1,1],...,”E”:[1,1,1,1,1],   
”best”:”A”,”worst”:”B”

The Claude Opus 5 judge is run at temperature 0 with returned reasoning disabled. Each image is resized to 448 × 448 and JPEG-encoded at quality 90. The runner requests a strict JSON schema and permits at most 320 output tokens.

Evidence-grounded human evaluation. Motivated by multi-dimensional visual-grounding evaluation (Li et al., 2026), two human evaluators assessed model responses on 100 ContextCanvas cases. Model identities were concealed. For each case, evaluators were shown the image, scenario, answer options, gold decision, expert decisive-cue rationale, and an anonymized response. They independently judged whether the response selected the correct answer, identified the decisive visible cue or a visually warranted equivalent, and correctly linked that cue to the stated use. Valid equivalent reasoning was accepted without requiring lexical overlap with the expert rationale. Disagreements were resolved through discussion. Evidence-Grounded Accuracy requires all three conditions.

## S2.3 Evaluation Rubrics and Metrics

## S2.3.1 Dataset-Quality Rubrics

For CritiqueCanvas, five human reviewers independently inspected a stratified sample of 1,000 retained instruction–response pairs spanning photography, painting, and virtual imagery. The audit assessed instruction consistency, visual grounding, image-specific aesthetic analysis, and overall coherence and usefulness. Blocking issues include an instruction–image mismatch, failure to perform the requested task, material visual hallucination, an essentially generic or incomplete response, or a safety, rights, or compliance failure. Non-blocking issues are localized weaknesses, such as minor imprecision, awkward wording, limited repetition, or omission of a secondary point, that do not by themselves invalidate the pair. Blocking issues require rejection, whereas non-blocking issues alone do not.

Of the 1,000 audited pairs, 970 received no blocking-issue flag from any of the five independent reviewers. The 97.0% “raw agreement” reported in the main paper denotes this unanimous no-blocking-issue rate, not a chance-corrected inter-rater reliability coeficient. A subsequent asset-level validation, conducted separately from response-quality voting, removed 17 otherwise acceptable pairs whose source images did not meet the final clarity threshold. The resulting 953 pairs constitute the reported 95.3% accepted set.

For ContextCanvas, four PhD-level researchers rechecked and adjudicated the candidate cases. A case was retained only when all seven criteria held: aesthetic centrality requires judgment of the image as a visual choice; context dependence requires the use, audience, or convention to afect the decision; visual necessity prevents reduction to text-only factual recall; decisive evidence requires the cultural, historical, narrative, or functional evidence to materially afect the answer; scenario naturalness requires a plausible design or communication use; answerability requires a uniquely supportable closed-form label; and anti-shortcut validity prevents wording, option length, or label priors from exposing the answer. Figure S3 supplies boundary examples for target dificulty, visual necessity, and answerability.

## S2.3.2 Critique Evaluation

We use BLEU (Papineni et al., 2002), ROUGE-L (Lin, 2004), and METEOR (Banerjee and Lavie, 2005) for lexical overlap; BERT-F1 (Zhang et al., 2020) and SBERT cosine similarity (Reimers and Gurevych, 2019) for token- and sentence-level semantic similarity; and a CLIP image–text cosine score (Hessel et al., 2021) for image relevance. Let $h _ { i }$ and $r _ { i }$ denote the generated and reference critiques for item i.

BLEU is SacreBLEU corpus BLEU (Post, 2018) with maximum order four and effective order=True. With clipped corpus precision $p _ { n }$ and brevity penalty $\mathrm { B P } = 1 \mathrm { i f } c > r ,$ otherwise $\exp ( 1 - r / c )$ , and SacreBLEU’s default exponential smoothing for zero higher-order counts, the percentage-form value in the main table is

$$
\mathrm { B L E U } = 1 0 0 \mathrm { B P } \exp \left( \frac { 1 } { 4 } \sum _ { n = 1 } ^ { 4 } \log p _ { n } \right) .\tag{S1}
$$

The evaluation output stores this value divided by 100; the paper multiplies it by 100 for presentation. ROUGE-L is computed per item with stemming enabled. If $L _ { i }$ is the longest-common-subsequence length, $P _ { i } = L _ { i } / | h _ { i } |$ , and $R _ { i } = L _ { i } / | r _ { i } |$ , then

$$
\mathrm { { R O U G E - L } } = \frac { 1 } { N } \sum _ { i } \frac { 2 P _ { i } R _ { i } } { P _ { i } + R _ { i } } ,\tag{S2}
$$

with a zero contribution when the denominator is zero.

METEOR uses NLTK’s default English matcher (Bird, Klein, and Loper, 2009) on whitespace-tokenized texts. For the matcher-selected unigram alignment with $m _ { i }$ matches and $c h _ { i }$ chunks,

$$
P _ { i } = m _ { i } / | h _ { i } | ,\tag{S3}
$$

$$
R _ { i } = m _ { i } / | r _ { i } | ,\tag{S4}
$$

$$
F _ { i } = \frac { P _ { i } R _ { i } } { 0 . 9 P _ { i } + 0 . 1 R _ { i } } ,\tag{S5}
$$

$$
\mathrm { P e n } _ { i } = 0 . 5 ( c h _ { i } / m _ { i } ) ^ { 3 } ,\tag{S6}
$$

$$
\mathrm { M E T E O R } = \frac { 1 } { N } \sum _ { i } ( 1 - \mathrm { P e n } _ { i } ) F _ { i } .\tag{S7}
$$

Items with no alignment contribute zero. The implementation also inherits NLTK’s exact, stemmed, and WordNet-based synonym matching behavior.

BERTScore uses microsoft/deberta-base-mnli, no IDF weighting, and no baseline rescaling. For contextual token embeddings, its item-level precision and recall are the mean maximum cosine matches from hypothesis to reference and reference to hypothesis; BERT-F1 is the corpus mean of $2 P i R _ { i } / ( P _ { i } + R _ { i } )$ .

SBERT-Cos uses sentence-transformers/all-mpnet-base-v2; masked mean pooling over the last hidden state is $\begin{array} { r } { e ( x ) = \sum _ { t } a _ { t } z _ { t } / \sum _ { t } a _ { t } } \end{array}$ , followed by $\ell _ { 2 }$ normalization and

$$
\mathrm { S B E R T - C o s } = \frac { 1 } { N } \sum _ { i } e ( h _ { i } ) ^ { \top } e ( r _ { i } ) .\tag{S8}
$$

Both text encoders truncate at 512 tokens.

The reported CLIP value uses openai/clip-vit-base-patch32. For normalized CLIP text and image features $t ( h _ { i } )$ and $v ( I _ { i } )$ ,

$$
\mathrm { C L I P C o s } = \frac { 1 } { N } \sum _ { i } t ( h _ { i } ) ^ { \top } \boldsymbol { v } ( I _ { i } ) .\tag{S9}
$$

This implementation is the raw mean cosine: it does not apply the $\operatorname* { m a x } ( \cdot , 0 )$ truncation or $2 . 5 \times$ scaling sometimes used by the named CLIPScore metric. We retain the main-table column name for continuity but make the operational definition explicit here. All six metrics are interpreted jointly because no single reference exhausts the valid analyses of an image.

For response-level statistics, text is lowercased and tokenized with the regular expression $\left[ { { \bf { a } } - { z } } \right] +$ lightweight Markdown headings and markers are removed before dimension matching. For generated response $\hat { r } _ { i }$ , average length and explicit dimension coverage are

$$
\mathrm { A v g L e n } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } | \operatorname { w o r d s } ( \hat { r } _ { i } ) | ,\tag{S10}
$$

$$
\mathrm { D i m H i t s } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \sum _ { d \in D } m ( \hat { \boldsymbol { r } } _ { i } , d ) ,\tag{S11}
$$

where $m ( \hat { r } _ { i } , d ) = 1$ only when response $\hat { r } _ { i }$ explicitly addresses dimension d. Top-3 Dimensions are those with the largest corpus-level $\Sigma _ { i } m ( \hat { r } _ { i } , d )$ . The ten categories and their fixed lexical/phrase triggers are Content, Composition, Color, Lighting, Brushstroke/Texture, Style, Emotion, Technique, Symbolism, and Visual Appeal. If $q ( \hat { r } _ { i } )$ counts all matched lexical/phrase occurrences and $F _ { i }$ and $M _ { i }$ are the prompt-focused and response-matched dimension sets, respectively, then

$$
\mathrm { D e n s i t y } = 1 0 0 \frac { \sum _ { i } q ( \hat { r } _ { i } ) } { \sum _ { i } | \operatorname { w o r d s } ( \hat { r } _ { i } ) | } ,\tag{S12}
$$

$$
\operatorname { P r o m p t A l i g n } = \frac { 1 } { N _ { f } } \sum _ { i : F _ { i } \neq \emptyset } \mathbf { 1 } [ F _ { i } \subseteq M _ { i } ] .\tag{S13}
$$

Thus density counts repeated mentions rather than merely distinct categories. Generic Top-3 excludes prompts in which the current question explicitly names an aesthetic dimension; few-shot examples are removed before focus detection. These lexical diagnostics measure verbosity, breadth, and controllability rather than correctness, relevance, or grounding.

The human critique-quality evaluation applies four binary criteria: visual grounding, aesthetic specificity, criterion appropriateness, and analytical usefulness. For critique $i ,$ evaluator $^ { \cdot } j ,$ and criterion $k ,$ the item-level score is

$$
H _ { i j } = 1 + \sum _ { k = 1 } ^ { 4 } b _ { i j k } , \qquad b _ { i j k } \in \{ 0 , 1 \} .\tag{S14}
$$

With two evaluators, Table 4 reports

$$
{ \mathrm { H u m a n S c o r e } } = { \frac { 1 } { 2 N } } \sum _ { i = 1 } ^ { N } \sum _ { j = 1 } ^ { 2 } H _ { i j } .\tag{S15}
$$

The MLLM judge assigns four criterion-specific scores and a fifth holistic overall quality score, each from 1 to 5. Table 4 in the main paper reports the mean of the fifth score:

$$
\mathrm { M L L M S c o r e } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } s _ { i } ^ { \mathrm { o v e r a l l } } .\tag{S16}
$$

The human and MLLM protocols for the 100-case critique-metric validity analysis, including anonymization, response permutation, rating criteria, and score aggregation, are specified in Section S2.2.3.

## S2.3.3 Contextual-Suitability Evaluation

Exact-match accuracy uses all 301 cases:

$$
\mathrm { A c c u r a c y } = { \frac { 1 } { 3 0 1 } } \sum _ { i = 1 } ^ { 3 0 1 } \mathbf { 1 } [ { \hat { y } } _ { i } = y _ { i } ] .\tag{S17}
$$

Macro-F1 and predicted Yes Rate are computed on the 291 two-option cases; the 10 three-option cases contribute only to Accuracy. The scorer operates on option IDs A and B. In 287 cases these are explicitly Yes and No; four cases use context-specific binary alternatives and retain the same A/B class mapping. The paper therefore reports A as the canonical Yes/target-aligned class and B as No/non-target class. For class $k \in \{ A , B \}$ ,

$$
P _ { k } = { \frac { T P _ { k } } { T P _ { k } + F P _ { k } } } , \quad R _ { k } = { \frac { T P _ { k } } { T P _ { k } + F N _ { k } } } , \quad F 1 _ { k } = { \frac { 2 P _ { k } R _ { k } } { P _ { k } + R _ { k } } } ,\tag{S18}
$$

and

$$
\mathrm { M a c r o F 1 } = \frac { F 1 _ { A } + F 1 _ { B } } { 2 } , \qquad \mathrm { Y e s R a t e } = \frac { 1 } { 2 9 1 } \sum _ { i } { \bf 1 } [ \hat { y } _ { i } = A ] .\tag{S19}
$$

The gold Yes Rate is 38.14%. Missing, conflicting, ambiguous, and unparseable answers are incorrect.

Evidence-Grounded Accuracy requires a correct label, identification of the decisive visible cue or a warranted equivalent, and a correct link from that cue to the stated use:

$$
\mathrm { E G A } = \frac { 1 } { N } \sum _ { i } \mathbf { 1 } [ \mathrm { c o r r e c t } _ { i } \wedge \mathrm { g r o u n d e d } _ { i } \wedge \mathrm { c o n t e x t L i n k } _ { i } ] .\tag{S20}
$$

Among correct predictions, the unsupported rate is

$$
\mathrm { U n s u p p o r t e d } = 1 0 0 \times { \frac { \# \{ { \mathrm { c o r r e c t ~ b u t ~ n o t ~ g r o u n d e d } } \} } { \# \{ { \mathrm { c o r r e c t } } \} } } .\tag{S21}
$$

For the 36 matched counterfactual pairs, Net Correct Update is

$$
\mathrm { N C U } = 1 0 0 \times { \frac { N _ { \mathrm { c o r r e c t ~ c h a n g e } } - N _ { \mathrm { r e v e r s e ~ c h a n g e } } } { 3 6 } } .\tag{S22}
$$

NCU is reported together with correct, reverse, unchanged-correct, and unchanged-wrong counts and original/edit accuracies; it is not interpreted as a standalone causal score. Bootstrap 95% confidence intervals (Efron and Tibshirani, 1993) are used for accuracy and diagnostic rates. Base–specialist comparisons use exact two-sided McNemar tests (McNemar, 1947) on paired predictions with Bonferroni correction (Dunn, 1961) across the four reported pairs. Descriptive case studies receive no significance claim.

## S2.4 Implementation Details

## S2.4.1 Model and Inference Configuration

All models receive the same task content—the image(s), record-level question or scenario, and fixed options where applicable—without web search, retrieval, reverse-image search, metadata access, or auxiliary recognition tools. Images are converted to RGB and resized or padded to 448 × 448 before being serialized with each checkpoint’s native multimodal chat template. This model-specific serialization does not alter the task content.

Evaluation uses deterministic decoding whenever supported. Local runs decode greedily with at most 256 new tokens. Closed-API runs use temperature 0, four concurrent requests, a 180-second timeout, and at most three transport retries. The ContextCanvas evaluator permits up to 800 output tokens for GPT-5.2 and Gemini 3.1 Pro and 500 for Qwen3.7-Plus. For the single paired-image item, checkpoints accepting only one image tensor receive a labeled horizontal contact sheet.

Table S3 summarizes the CritiqueCanvas inference configurations used in evaluation.

Table S3: CritiqueCanvas inference configurations. “Native” means the checkpoint’s own processor and chat template after the common 448 × 448 RGB resize.
<table><tr><td>Model</td><td>Requested model/checkpoint</td><td>Precision</td><td>Generation</td><td>Adapter details</td></tr><tr><td>GPT-5.2</td><td>openai/gpt-5.2</td><td>API</td><td>T = 0, 256</td><td>OpenRouter; reasoning excluded</td></tr><tr><td>Gemini 3.1 Pro</td><td>google/gemini-3.1-pro-preview</td><td>API</td><td>T = 0, 256</td><td>Google provider preference; minimal reasoning ex- cluded</td></tr><tr><td>GLM-4.6-Flash</td><td>GLM-4.6-Flash</td><td>bfloat16</td><td>greedy, 256</td><td>Native; thinking disabled and stripped</td></tr><tr><td>LLaVA-OneVision</td><td>LLaVA-OneVision-1.5-8B-Instruct</td><td>float16</td><td>greedy, 256</td><td>Native</td></tr><tr><td>InternVL3.5-8B</td><td>InternVL3_5-8B</td><td>float16</td><td>greedy, 256</td><td>Native InternVL chat API</td></tr><tr><td>Qwen3-VL</td><td>Qwen3-VL-8B-Instruct</td><td>bfloat16</td><td>greedy, 256</td><td>Qwen processor; SDPA</td></tr><tr><td>Qwen3-VL-FT</td><td>Qwen3-VL-8B + CritiqueCanvas LoRA</td><td>bfloat16</td><td>greedy, 256</td><td>PEFT adapter loaded without merge; SDPA</td></tr><tr><td>ArtQuant-APDD</td><td>mPLUG-Owl2-LLaMA2-7B + APDD</td><td>default</td><td>greedy, 256</td><td>Official ArtQuant loader; no 4/8-bit quantization</td></tr><tr><td>ArtiMuse</td><td>ArtiMuse</td><td>bfloat16</td><td>greedy, 256</td><td>Official InternVL-family loader; FlashAttention off</td></tr><tr><td>UniPercept</td><td>UniPercept</td><td>bfloat16</td><td>greedy, 256</td><td>Official InternVL-style chat; FlashAttention off</td></tr><tr><td>AesExpert</td><td>AesMMIT_LLaVA_v1.5_7b_240325</td><td>default</td><td>greedy, 256</td><td>Official LLaVA loader; 11ava_v1 conversation</td></tr><tr><td>Q-SiT</td><td>q-sit</td><td>float16</td><td>greedy, 256</td><td>LLaVA-OneVision generation API</td></tr></table>

## S2.4.2 Fine-Tuning Configuration

Qwen3-VL-FT is adapted on CritiqueCanvas and evaluated on critique generation in the main paper. At evaluation time, the resulting PEFT adapter is loaded onto Qwen3-VL-8B-Instruct without merging. Evaluation uses bfloat16 precision and SDPA, applies the base processor’s multimodal chat template to the image and record question, resizes the image to $4 4 8 \times 4 4 8 .$ , and decodes greedily for at most 256 new tokens.

## S2.4.3 Output Processing and Aggregation

The ContextCanvas parser first accepts schema-valid JSON and verifies that the answer belongs to the supplied option IDs. Narrow compatibility fallbacks accept one unambiguous answer field, a single leading option ID, or an exact Yes/No string for specialist outputs. Rationales and confidence never change the grade. The evaluator permits at most four transport attempts; retries recover a missing transport or model response and never allow revision of a valid answer.

The evaluation pipeline performs answer parsing and computes Accuracy, Macro-F1, Yes Rate, counterfactual directional counts, EGA, and the reported critique metrics from prediction files. Its output schema and aggregation rules are specified above.

## S3 Additional Analyses and Insights

This section complements the main tables with response-level evidence and paired diagnostics. We first examine acceptance tendency and decisive-cue grounding, then combine visual counterfactuals, modality decomposition, and base–specialist comparisons to distinguish surface aesthetic fluency, visual evidence extraction, and context-sensitive judgment.

## S3.1 Model-Output Analysis

## S3.1.1 Aggregate Behavioral Patterns

The main ContextCanvas table shows diferences not only in exact-match accuracy but also in the direction of errors. Against a 38.14% gold Yes rate, the four aesthetic-specific models predict Yes for 59.11–76.98% of the 291 two-option cases. The strongest closed-source models instead predict Yes for 20.96–30.24% of cases. These rates expose a systematic diference in decision tendency: aesthetic specialists more often accept images whose formal or surface qualities appear plausible despite a contextual mismatch.

Table S4 expands the main results into class-specific F1. The strongest closed-source models remain efective on both labels, whereas several open-weight and aesthetic-specific models have very low No-class F1. Q-SiT is the clearest example: its 76.98% Yes rate is accompanied by only 4.05 No-class F1. Conversely, GPT-5.2 is conservative, with a 20.96% Yes rate and a much larger gap between Yes- and No-class F1. Thus similar overall scores can conceal diferent failure profiles.

Table S4: Class-specific behavior on the 291 two-option ContextCanvas cases. All values are percentages. The gold Yes rate is 38.14%.
<table><tr><td>Model</td><td>Yes F1</td><td>No F1</td><td>Pred. Yes</td></tr><tr><td>Claude Opus 5</td><td>87.44</td><td>93.47</td><td>30.24</td></tr><tr><td>Gemini 3.1 Pro</td><td>77.01</td><td>89.11</td><td>26.12</td></tr><tr><td>GPT-5.5</td><td>73.30</td><td>86.96</td><td>27.49</td></tr><tr><td>Claude Opus 4.6</td><td>56.84</td><td>79.08</td><td>27.15</td></tr><tr><td>GPT-5.2</td><td>50.00</td><td>79.02</td><td>20.96</td></tr><tr><td>GLM-5V-Turbo</td><td>47.06</td><td>63.37</td><td>43.64</td></tr><tr><td>Qwen3.7-Plus</td><td>37.93</td><td>58.86</td><td>41.58</td></tr><tr><td>Grok 4.20</td><td>24.43</td><td>53.74</td><td>37.80</td></tr><tr><td>Qwen3-VL-Instruct</td><td>28.22</td><td>49.27</td><td>44.67</td></tr><tr><td>InternVL3-8B</td><td>21.92</td><td>21.38</td><td>62.20</td></tr><tr><td>mPLUG-Owl2</td><td>7.78</td><td>27.08</td><td>50.17</td></tr><tr><td>LLaVA-OneVision</td><td>25.79</td><td>10.61</td><td>71.13</td></tr><tr><td>LLaVA-v1.5</td><td>9.06</td><td>11.53</td><td>60.48</td></tr><tr><td>ArtQuant-APDD</td><td>27.56</td><td>31.44</td><td>59.11</td></tr><tr><td>ArtiMuse</td><td>18.43</td><td>17.30</td><td>62.54</td></tr><tr><td>Q-SiT</td><td>29.25</td><td>4.05</td><td>76.98</td></tr><tr><td>AesExpert</td><td>20.33</td><td>12.27</td><td>66.67</td></tr></table>

Prediction tendency does not by itself reveal whether the rationale uses the decisive image evidence. On the 100-case grounding audit, Gemini retains 79 of 83 correct decisions under EGA, while GPT-5.2 retains 55 of 65 and Qwen3.7-Plus retains 31 of 42. ArtiMuse falls from 21 correct labels to only three answers that are also grounded and correctly linked to the use scenario. The Accuracy-to-EGA reduction therefore complements predicted Yes rate by separating label tendency from grounded success.

## S3.1.2 Cross-Model Case Studies

Cultural-aesthetic alignment. Figure S4 compares five models on HB208. The image has an attractive moon-centered composition, but its characters and visual language are tied to Japanese Touhou fan culture rather than an unqualified Chinese Mid-Autumn identity. Gemini 3.1 Pro and GPT-5.2 reject the image. Qwen3.7-Plus instead maps the full moon, tea, bamboo, and rabbits directly to Chinese Mid-Autumn symbolism. ArtiMuse repeats the positive formal framing and accepts it. ArtQuant-APDD also predicts Yes, accompanied only by the title-like phrase “A Serene Moment in a Chinese Landscape,” which misidentifies the cultural setting. The disagreement is therefore not whether the image is appealing, but whether that aesthetic reading is culturally valid for the stated use.

Aesthetic valence versus communicative function. Figure S5 compares five models on HB181, which tests whether models can distinguish pleasant appearance from efective visual rhetoric. Several models reject the painting because its rough water, dark tonal range, tilted boat, and forceful diagonal net produce instability and tension rather than a conventionally uplifting image. Yet these formal qualities intensify the visible coordination of the two workers: their aligned bodies and shared control of the net make cooperation legible precisely under pressure. The case shows that contextual aesthetic suitability may depend on how composition, tone, and visual tension serve a communicative purpose, rather than on positive surface afect alone.

![](images/8fc0725c308c8be46af0e7e2ffb9cf53b7443ba54e4b16d23a7eaff588f2b096.jpg)  
Figure S4: Cross-model outputs for ContextCanvas case HB208 (gold: No). GPT-5.2 and Gemini 3.1 Pro reject the image, whereas Qwen3.7-Plus, ArtiMuse, and ArtQuant-APDD accept it. The displayed responses illustrate whether models recognize the culturally specific visual identity rather than relying only on surface festival cues.

Critique quality. Figure S6 supplies a complementary generation-side comparison. Valid critiques of the same image may emphasize diferent visual evidence or aesthetic criteria, so low overlap with one reference does not necessarily imply low practical quality. The three selected cases contrast grounded image-specific analysis with hallucinated subjects, medium-inappropriate criteria, and fluent but generic interpretations.

## S3.1.3 Failure Modes and Representative Outputs

The qualitative outputs instantiate the four failure levels introduced in the main paper. Fluent but visually ungrounded critiques invent objects or properties absent from the image. Domain-blind quality priors apply a photographic defect vocabulary, such as blur or noise, to intentional painterly texture. Surface aesthetics over communicative fit treats attractive or unattractive afect as suficient evidence of suitability. Stylistic plausibility over contextual fit accepts an image because its palette, composition, or period atmosphere appears coherent while overlooking the identity, convention, or narrative that determines the use.

The ContextCanvas casebook further illustrates three recurrent subtypes without claiming corpus-wide frequencies. In cultural-neighbor substitution, a nearby visual convention is mapped to the wrong occasion or identity (HB208, HB014). In aesthetic-halo errors, refinement or spectacle is treated as proof of contextual appropriateness (HB037, HB186). In salient-composition myopia, a dominant formal cue suppresses a smaller but decisive action or relation (HB106, HB181). These labels summarize reviewed examples; a frequency claim would require an independent coding pass over all 301 cases.

Figure S7 distinguishes a correct grounded response, a correct but unsupported label, a fluent but

![](images/a9b5402292533442214a4b6986bcaf3e7fc0185996e7bd6ec8883e786c421584.jpg)  
Figure S5: ArtQuant-APDD returns a bare Yes response without an explanatory rationale;

incorrect rationale, and a free-form response illustrating output-format sensitivity.

## S3.2 Extended and Robustness Analyses

## S3.2.1 Extended Main Results

The binary error decomposition in Table S4 explains why Macro-F1 is lower than exact-match accuracy for many models: the gold set is No-majority, while several models disproportionately predict Yes. In particular, matching the marginal gold Yes rate is not suficient—Grok’s 37.80% predicted Yes rate is close to 38.14%, but its Macro-F1 is only 39.09%. Contextual competence depends on instance-level alignment rather than calibration of the marginal label count.

The main paper also promises response-level statistics beyond reference-matching metrics. Table S5 reports output length, average dimension coverage, dimension density, adherence to dimension-focused prompts, and the most frequent dimensions in otherwise generic critiques. As defined in Section S2.3, these are descriptive measures of response form and explicit coverage rather than direct measures of correctness or critique quality.

The statistics expose three complementary patterns. First, response length is not a proxy for useful aesthetic analysis: Qwen3-VL produces the longest critiques (199.07 words under the evaluation tokenizer), whereas UniPercept is substantially shorter (102.12) but has the highest dimension density (13.26). Second, breadth and controllability vary independently. Gemini obtains the broadest dimension coverage (6.53 dimensions), while Qwen3-VL has the strongest focused-prompt alignment (95.2%); by contrast, ArtQuant and AesExpert follow such prompts much less reliably (66.7% and 34.4%). Third, generic critiques concentrate heavily on composition, content, and emotion across model families. Lighting, color, and brushstrokes appear among the top dimensions for only a few models, while technique, symbolism, and style never enter the generic top three. Thus, models difer markedly in verbosity and response concentration, but their open-ended critiques still rely on a relatively narrow set of salient aesthetic concepts. These patterns complement—rather than replace—the direct quality audit below, since greater length or dimension coverage does not establish that the corresponding claims are relevant, correct, or visually grounded.

![](images/c910f035199949c05b0ed177a4b734d41f323e2d3d02ab85e8fdd7e4f2763c74.jpg)  
Figure S6: Three CritiqueCanvas cases compared across five models. From top to bottom: CC001 (illustration), CC003 (photograph), and CC002 (painting). The panel preserves each prompt and representative output excerpt, exposing diferences in visual grounding, medium specificity, and analytical focus.

## S3.2.2 Critique Metric Validity

Using the 100-case audit protocol described in Section S2.3.2, Table S6 reproduces the aggregate direct ratings reported in the main paper.

Table S6: Aggregate human and MLLM ratings of critique quality on 100 randomly sampled CritiqueCanvas cases.
<table><tr><td>Model</td><td>Human</td><td>MLLM judge</td></tr><tr><td>GPT-5.2</td><td>3.9</td><td>5.0</td></tr><tr><td>Gemini 3.1 Pro</td><td>4.1</td><td>4.5</td></tr><tr><td>Qwen3-VL-FT</td><td>2.5</td><td>2.5</td></tr><tr><td>ArtiMuse</td><td>2.6</td><td>2.6</td></tr><tr><td>ArtQuant-APDD</td><td>2.3</td><td>2.2</td></tr></table>

The human ordering is consistent with the complementary judge at the group level: GPT-5.2 and Gemini receive higher direct-quality ratings than the adapted or aesthetic-specific models, although they do not dominate all reference-similarity metrics. This supports a narrow conclusion. Lexical and semantic reference alignment captures similarity to one expert answer but only partially reflects grounding, image specificity, criterion choice, and practical usefulness when multiple analyses are defensible. It does not show that

![](images/331814ea8c0bb9cd248ad2ab86277cd351b9b8e1fc81d375495f1d32cb762fd3.jpg)  
Figure S7: Response-level scoring diagnostics on E004 (top) and HB208 (bottom). E004 contrasts a grounded GPT-5.2 rationale with an unsupported ArtQuant-APDD label. HB208 contrasts fluent but incorrect Qwen3.7-Plus reasoning with an ArtQuant-APDD free-form response that illustrates output-format and parser sensitivity.

BLEU, SBERT, or CLIPScore is generally invalid. Figure S6 provides illustrative examples of this mismatch: critiques with lower reference similarity can remain visually grounded and analytically useful, whereas closer stylistic alignment does not necessarily prevent generic analysis or unsupported visual claims.

## S3.2.3 Counterfactual and Grounding Analyses

Visual counterfactual sensitivity. Controlled visual counterfactuals can diagnose reliance on decisive image evidence (Chen et al., 2020). We construct 36 matched No-to-Yes pairs by replacing a decisive incompatible visual cue with a scenario-compatible alternative while keeping the use scenario fixed.

Table S7 reports original and edited accuracy, Pair Accuracy, directional updates, and the NCU defined in Section S2.3.3. Pair Accuracy requires both the incompatible original and compatible edit to be classified correctly.

Table S7: Counterfactual audit on 36 pairs. Full and Orig. are accuracies on the complete benchmark and selected incompatible originals; Edit is accuracy on compatible edits. Pair requires both labels to be correct. C/R are correct and reverse directional updates. NCU confidence intervals are paired bootstrap intervals.
<table><tr><td>Model</td><td>Full</td><td>Orig.</td><td>Edit</td><td>Pair</td><td>C/R</td><td>NCU</td><td>95% CI</td></tr><tr><td>GPT-5.2</td><td>71.43</td><td>72.22</td><td>58.33</td><td>36.11</td><td>13/2</td><td>30.56</td><td>[11.11, 50.00]</td></tr><tr><td>Gemini 3.1 Pro</td><td>85.71</td><td>86.11</td><td>38.89</td><td>27.78</td><td>10/1</td><td>25.00</td><td>[8.33, 41.67]</td></tr><tr><td>Qwen3.7-Plus</td><td>51.83</td><td>52.78</td><td>75.00</td><td>30.56</td><td>11/1</td><td>27.78</td><td>[11.11, 44.44]</td></tr><tr><td>ArtQuant-APDD</td><td>29.57</td><td>30.56</td><td>72.22</td><td>8.33</td><td>3/2</td><td>2.78</td><td>[-8.33, 13.89]</td></tr><tr><td>ArtiMuse</td><td>19.60</td><td>19.44</td><td>97.22</td><td>16.67</td><td>6/0</td><td>16.67</td><td>[5.56, 30.56]</td></tr></table>

Table S5: Generation style and aesthetic-dimension coverage on CritiqueCanvas. Len/Ref is average output length relative to the reference; DimHits is the average number of explicitly addressed dimensions; aesthetic-term density is the number of matched lexical/phrase occurrences per 100 words. Prompt Align requires every dimension detected in a focused prompt to be detected in the response. Generic Top 3 excludes prompts that explicitly request a dimension.
<table><tr><td>Model</td><td></td><td></td><td></td><td></td><td></td><td>Length Len/Ref DimHits Density Prompt Align Generic Top 3 Dimensions</td></tr><tr><td colspan="7">Closed-source Models</td></tr><tr><td>Gemini 3.1 Pro</td><td>166.58</td><td>1.241</td><td>6.53</td><td>11.87</td><td>90.8%</td><td>Composition, Emotion, Lighting</td></tr><tr><td>GPT-5.2</td><td>141.52</td><td>1.054</td><td>5.71</td><td>8.60</td><td>93.8%</td><td>Composition, Emotion, Lighting</td></tr><tr><td colspan="7">General MLLMs</td></tr><tr><td>GLM-4.6-Flash</td><td>189.37</td><td>1.410</td><td>6.50</td><td>9.44</td><td>94.2%</td><td>Composition, Content, Emotion</td></tr><tr><td>InternVL3.5-8B</td><td>142.78</td><td>1.090</td><td>5.17</td><td>11.18</td><td>92.5%</td><td>Content, Composition, Emotion</td></tr><tr><td>LLaVA-OneVision</td><td>88.97</td><td>0.679</td><td>4.12</td><td>10.60</td><td>79.8%</td><td>Content, Composition, Emotion</td></tr><tr><td>Qwen3-VL</td><td>199.07</td><td>1.520</td><td>5.54</td><td>9.29</td><td>95.2%</td><td>Composition, Content, Emotion</td></tr><tr><td>Qwen3-VL-FT</td><td>120.74</td><td>0.922</td><td>4.45</td><td>10.96</td><td>93.3%</td><td>Composition, Content, Emotion</td></tr><tr><td colspan="7">Aesthetic Models</td></tr><tr><td>ArtQuant</td><td>103.86</td><td>0.793</td><td>3.99</td><td>9.95</td><td>66.7%</td><td>Content, Emotion, Composition</td></tr><tr><td>ArtiMuse</td><td>100.07</td><td>0.764</td><td>4.95</td><td>12.54</td><td>90.8%</td><td>Composition, Content, Emotion</td></tr><tr><td>UniPercept</td><td>102.12</td><td>0.780</td><td>5.15</td><td>13.26</td><td>92.3%</td><td>Composition, Content, Emotion</td></tr><tr><td>AesExpert</td><td>35.68</td><td>0.272</td><td>2.34</td><td>10.34</td><td>34.4%</td><td>Composition, Lighting, Color</td></tr><tr><td>Q-SiT</td><td>82.03</td><td>0.626</td><td>3.63</td><td>9.23</td><td>75.4%</td><td>Composition, Content, Brushstrokes</td></tr></table>

The three general-purpose models occupy a narrow 25.00–30.56 NCU range despite substantially diferent original accuracy. ArtQuant changes its decision on only five pairs, with three correct and two reverse updates, producing an NCU of 2.78 whose interval includes zero. ArtiMuse makes six correct and no reverse updates, but predicts Yes for 64 of the 72 original/edit images. Its 97.22% edited accuracy therefore largely reflects an acceptance tendency; Pair Accuracy and NCU provide the more discriminative diagnosis. The mean NCU is 27.78 for the three general-purpose models and 9.72 for the two specialists.

The rationale audit additionally reveals source-scene anchoring: some responses repeat a canonical narrative even after its supporting visual cue is removed.

Figure S8 presents representative interventions with each model’s before–after prediction

Evidence-grounded accuracy. Using the EGA and Unsupported definitions in Section S2.3.3, Table S8 expands the 100-case audit.

Table S8: Decisive-cue grounding on 100 ContextCanvas cases.
<table><tr><td>Model</td><td>Accuracy</td><td>EGA</td><td>Unsupported (%)</td></tr><tr><td>Gemini 3.1 Pro</td><td>83</td><td>79</td><td>4.82</td></tr><tr><td>GPT-5.2</td><td>65</td><td>55</td><td>15.38</td></tr><tr><td>Qwen3.7-Plus</td><td>42</td><td>31</td><td>26.19</td></tr><tr><td>ArtiMuse</td><td>21</td><td>3</td><td>85.71</td></tr><tr><td>ArtQuant-APDD</td><td>27</td><td>0</td><td>100.00</td></tr></table>

![](images/357bfb3b8a7f045886d5599e70d327f7d4d48a71838ec8214773b535c5f2f8cc.jpg)  
Figure S8: Four No-to-Yes counterfactual interventions. From left to right: CF20/HB163 (coercion to reassurance), CF18/HB084 (isolation to inclusion), CF28/E108 (incomplete to completed journey), and CF35/HB075 (incompatibility to friendliness). Each column shows the original and edited image, the targeted intervention, and each model’s before–after prediction; colored boxes indicate agreement or disagreement with the corresponding gold label.

ArtiMuse is the cleaner specialist diagnosis because it always generates a substantive rationale: its fall from 21 correct labels to 3 grounded correct answers cannot be attributed to missing explanation text. Its prose is often fluent and aesthetically framed, but it seldom identifies and correctly binds the culture- or context-bearing cue. The audit therefore addresses the concern that binary accuracy alone may reward lucky labels or unsupported agreement.

Perception–reasoning decomposition. Following the view of captions as task-facing visual surrogates (Yang et al., 2026), Table S9 evaluates a 100-case diagnostic subset distinct from the 100 cases used for the evidence-grounding audit. Four conditions are compared: question and options only (Text); an identityand verdict-free description of visible content (Neutral); the same description with the decisive visible cue made explicit (Evidence); and the original image (Image).

Table S9: Accuracy under diferent visual-information conditions on 100 ContextCanvas cases.
<table><tr><td>Model</td><td>Text</td><td>Neutral</td><td>Evidence</td><td>Image</td></tr><tr><td>GPT-5.2</td><td>45</td><td>77</td><td>84</td><td>71</td></tr><tr><td>Gemini 3.1 Pro</td><td>58</td><td>86</td><td>90</td><td>86</td></tr><tr><td>Qwen3.7-Plus</td><td>44</td><td>78</td><td>88</td><td>52</td></tr><tr><td>ArtQuant-APDD</td><td>16</td><td>20</td><td>19</td><td>27</td></tr><tr><td>ArtiMuse</td><td>16</td><td>32</td><td>47</td><td>20</td></tr></table>

Text-only performance does not exceed the 61% majority baseline. Supplying a neutral visual description improves the three general-purpose models by 28–34 points; for each model, the paired exact two-sided McNemar test gives $p < 1 0 ^ { - 7 }$ . This confirms that the judgments depend materially on visual content. The modality gap is model-specific: Gemini matches its image accuracy from a neutral description, whereas Qwen rises from 52% with images to 78% with neutral descriptions and 88% when the decisive cue is explicit. ArtiMuse is partly rescued by explicit evidence but remains far below the general-purpose models; ArtQuant remains weak across all conditions. Together with the counterfactual and EGA results, these conditions provide a diagnostic of whether errors are more consistent with limited visual-evidence extraction or with failure to use explicitly supplied contextual evidence; they are not a definitive causal decomposition.

## S3.2.4 Base-to-Specialist Transfer

Table S10 compares four aesthetic specialists with the base checkpoint identified by their implementation and model documentation. Because every pair is evaluated on the same 301 cases, we report the two discordant counts, a paired bootstrap interval for the accuracy change, an exact two-sided McNemar test, and the Bonferroni-adjusted value across four lineages.

Table S10: Paired base-to-specialist ContextCanvas transfer. b counts cases correct only for the base and c cases correct only for the specialist. Raw McNemar p-values and Bonferroni-adjusted values are both shown.
<table><tr><td>Base → specialist</td><td>Base</td><td>Spec.</td><td> $\Delta$ </td><td> $b / c$ </td><td>95% CI</td><td> $p / p _ { \mathrm { a d j } }$ </td></tr><tr><td>mPLUG-Owl2 → ArtQuant</td><td>20.27</td><td>29.57</td><td>+9.30</td><td>29/57</td><td>[3.32, 15.28]</td><td>.0034/.0134</td></tr><tr><td>InternVL3-8B → ArtiMuse</td><td>23.26</td><td>19.60</td><td>-3.65</td><td>27/16</td><td>[-7.97, 0.33] .1263/.5052</td><td></td></tr><tr><td>LLaVA-v1.5 → AesExpert</td><td>12.62</td><td>18.27</td><td>+5.65</td><td>10/27</td><td>[1.66, 9.63] .0076/.0305</td><td></td></tr><tr><td>LLaVA-OV → Q-SiT</td><td>19.93</td><td>19.60</td><td>-0.33</td><td>12/11</td><td>[-3.32, 2.66] 1.000/1.000</td><td></td></tr></table>

Specialization has a heterogeneous efect. ArtQuant and AesExpert significantly improve over their corresponding bases after correction, while ArtiMuse and Q-SiT show no measurable ContextCanvas gain. Thus, specialization does not produce consistent gains in contextual judgment.

## References

Banerjee, S.; and Lavie, A. 2005. METEOR: An Automatic Metric for MT Evaluation with Improved Correlation with Human Judgments. In Goldstein, J.; Lavie, A.; Lin, C.-Y.; and Voss, C., eds., Proceedings of the ACL Workshop on Intrinsic and Extrinsic Evaluation Measures for Machine Translation and/or Summarization, 65–72. Ann Arbor, Michigan: Association for Computational Linguistics.

Bird, S.; Klein, E.; and Loper, E. 2009. Natural Language Processing with Python. O’Reilly Media.

Cao, S.; Ma, N.; Li, J.; Li, X.; Shao, L.; Zhu, K.; Zhou, Y.; Pu, Y.; Wu, J.; Wang, J.; Qu, B.; Wang, W.; Qiao, Y.; Yao, D.; and Liu, Y. 2025. ArtiMuse: Fine-Grained Image Aesthetics Assessment with Joint Scoring and Expert-Level Understanding. arXiv:2507.14533.

Chen, D.; Chen, R.; Zhang, S.; Liu, Y.; Wang, Y.; Zhou, H.; Zhang, Q.; Wan, Y.; Zhou, P.; and Sun, L. 2024. MLLM-as-a-Judge: Assessing Multimodal LLM-as-a-Judge with Vision-Language Benchmark. arXiv preprint arXiv:2402.04788.

Chen, L.; Yan, X.; Xiao, J.; Zhang, H.; Pu, S.; and Zhuang, Y. 2020. Counterfactual Samples Synthesizing for

Robust Visual Question Answering. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 10800–10809.

Dunn, O. J. 1961. Multiple Comparisons among Means. Journal of the American Statistical Association, 56(293): 52–64.

Efron, B.; and Tibshirani, R. J. 1993. An Introduction to the Bootstrap. Chapman and Hall/CRC.

Hessel, J.; Holtzman, A.; Forbes, M.; Le Bras, R.; and Choi, Y. 2021. CLIPScore: A Reference-free Evaluation Metric for Image Captioning. In Moens, M.-F.; Huang, X.; Specia, L.; and Yih, S. W.-t., eds., Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, 7514–7528. Online and Punta Cana, Dominican Republic: Association for Computational Linguistics.

Huang, Y.; Sheng, X.; Yang, Z.; Yuan, Q.; Duan, Z.; Chen, P.; Li, L.; Lin, W.; and Shi, G. 2024a. AesExpert: Towards Multi-modality Foundation Model for Image Aesthetics Perception. arXiv:2404.09624.

Huang, Y.; Yuan, Q.; Sheng, X.; Yang, Z.; Wu, H.; Chen, P.; Yang, Y.; Li, L.; and Lin, W. 2024b. Aes-Bench: An Expert Benchmark for Multimodal Large Language Models on Image Aesthetics Perception. arXiv:2401.08276.

Kong, S.; Shen, X.; Lin, Z.; Mech, R.; and Fowlkes, C. C. 2016. Photo Aesthetics Ranking Network with Attributes and Content Adaptation. In Computer Vision – ECCV 2016, volume 9905 of Lecture Notes in Computer Science, 662–679. Springer.

Li, C.; Zhang, Z.; Wu, H.; Sun, W.; Min, X.; Liu, X.; Zhai, G.; and Lin, W. 2024. AGIQA-3K: An Open Database for AI-Generated Image Quality Assessment. IEEE Transactions on Circuits and Systems for Video Technology, 34(8): 6833–6846.

Li, R.; Li, L.; Ren, S.; Tian, H.; Gu, S.; Li, S.; Yue, Z.; Wang, Y.; Ma, W.; Yang, Z.; Ma, J.; Sui, Z.; and Luo, F. 2026. GroundingME: Exposing the Visual Grounding Gap in MLLMs through Multi-Dimensional Evaluation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2412–2422.

Lin, C.-Y. 2004. ROUGE: A Package for Automatic Evaluation of Summaries. In Text Summarization Branches Out, 74–81. Barcelona, Spain: Association for Computational Linguistics.

Liu, H.; Huang, N.; Liu, C.; Yan, J.; Huang, H.; Ying, J.; Lee, T.-Y.; Wan, P.; and Ji, X. 2026. Bridging Cognitive Gap: Hierarchical Description Learning for Artistic Image Aesthetics Assessment. arXiv:2512.23413.

Liu, Y.; Iter, D.; Xu, Y.; Wang, S.; Xu, R.; and Zhu, C. 2023. G-Eval: NLG Evaluation Using GPT-4 with Better Human Alignment. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, 2511–2522.

McNemar, Q. 1947. Note on the Sampling Error of the Diference between Correlated Proportions or Percentages. Psychometrika, 12(2): 153–157.

Murray, N.; Marchesotti, L.; and Perronnin, F. 2012. AVA: A Large-Scale Database for Aesthetic Visual Analysis. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition.

Papineni, K.; Roukos, S.; Ward, T.; and Zhu, W.-J. 2002. Bleu: a Method for Automatic Evaluation of Machine Translation. In Isabelle, P.; Charniak, E.; and Lin, D., eds., Proceedings of the 40th Annual Meeting of the Association for Computational Linguistics, 311–318. Philadelphia, Pennsylvania, USA: Association for Computational Linguistics.

Post, M. 2018. A Call for Clarity in Reporting BLEU Scores. In Proceedings of the Third Conference on Machine Translation: Research Papers, 186–191.

Qi, D.; Zhao, H.; Shi, J.; Jenni, S.; Fan, Y.; Dernoncourt, F.; Cohen, S.; and Li, S. 2025. The Photographer Eye: Teaching Multimodal Large Language Models to See and Critique like Photographers. arXiv preprint arXiv:2509.18582.

Reimers, N.; and Gurevych, I. 2019. Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. In Inui, K.; Jiang, J.; Ng, V.; and Wan, X., eds., Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), 3982–3992. Hong Kong, China: Association for Computational Linguistics.

Wang, J.; Duan, H.; Liu, J.; Chen, S.; Min, X.; and Zhai, G. 2023. AIGCIQA2023: A Large-Scale Image Quality Assessment Database for AI Generated Images: From the Perspectives of Quality, Authenticity and Correspondence. In Artificial Intelligence: Third CAAI International Conference (CICAI 2023), volume 14474 of Lecture Notes in Computer Science, 46–57. Springer.

Yang, S.; Liu, Y.; Zhai, B.; Sun, X.; Liu, Z.; Barsoum, E.; Li, M.; and Xu, C. 2026. CaptionQA: Is Your Caption as Useful as the Image Itself? In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 23741–23750.

Yang, Y.; Xu, L.; Li, L.; Qie, N.; Li, Y.; Zhang, P.; and Guo, Y. 2022. Personalized Image Aesthetics Assessment with Rich Attributes. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 19861–19869.

Zhang, T.; Kishore, V.; Wu, F.; Weinberger, K. Q.; and Artzi, Y. 2020. BERTScore: Evaluating Text Generation with BERT. arXiv:1904.09675.