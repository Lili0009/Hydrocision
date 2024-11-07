AX='water_supply'
AW='nrwv_percentage'
AR='center'
AQ='manila_water_data.csv'
AV='Water Level (m)'
AP='%A %d %B, %Y  %I:%M%p'
AL='Up-Katipunan'
AK='rgba(255, 255, 255, 0.2)'
AJ='resetScale2d'
AI='coerce'
AH='water_data.csv'
AB='Timog'
AA='Tandang sora'
A9='San Juan'
A8='Elliptical'
A7='Araneta-Libis'
A6='solid'
A5='rgba(0, 0, 0, 0.7)'
A4='x unified'
A3='left'
A2='%b %d, %Y'
A1='Forecasted'
A0='Actual'
z='#7CFC00'
y=range
v=None
u='orange'
q='w'
p='Arial'
o='Bill Volume'
n='top'
l='nrwv'
k='lasso2d'
j='select2d'
i='lasso'
h='zoom'
g='Rainfall'
f=list
e=len
d=int
b='markers+lines'
a='displayModeBar'
Z='displaylogo'
Y='h'
X='Helvetica'
W='Business Zone'
V=False
U='%d-%b-%y'
T=round
S='r'
R=abs
O='Drawdown'
M='rgba(0,0,0,0)'
K='utf-8'
J=open
I='Supply Volume'
H='Water Level'
E=True
D='white'
C='Date'
A=dict
from keras.models import load_model as AC
from django.shortcuts import render as AD
import pandas as B,numpy as F
from sklearn.preprocessing import MinMaxScaler as AY
from sklearn.preprocessing import StandardScaler as AS
import datetime as AM,csv
from datetime import datetime as AN
import plotly.graph_objects as G,plotly.io as c
from django.http import JsonResponse as w
from django.templatetags.static import static
from django.utils import timezone as AO
from django.contrib.auth.decorators import login_required


def AT():
	global r,L,P,Q,r,AE,s;global x,m,AF,t,AZ,AU;x=AC('Model_water.h5');A=B.read_csv(AH);A[g]=B.to_numeric(A[g],errors=AI);D=d(e(A)*.8);G=A.iloc[:D];R=G[H].mean();S=G[O].mean();P=A.fillna(value={H:R,g:0,O:S}).copy();P[C]=B.to_datetime(P[C],format=U);P.set_index(C,inplace=E);m=AY(feature_range=(0,1));I=m.fit_transform(P)
	def J(data,seq_length):
		B=seq_length;A=data;D=[];E=[]
		for C in y(e(A)-B):D.append(A[C:C+B,:]);E.append(A[C+B,0])
		return F.array(D),F.array(E)
	K=30;AF,AZ=J(I[:D],K);t,AU=J(I[D:],K);T=f(P.index);V=15;M=480;W=B.date_range(f(T)[-V],periods=M,freq='d').tolist();X=x.predict(t[-M:]);Y=F.repeat(X,P.shape[1],axis=-1);Z=m.inverse_transform(Y)[:,0];s=[]
	for a in W:s.append(a.date())
	Q=B.DataFrame({C:F.array(s),H:Z});Q.set_index(C,inplace=E);L=P[[H]];L=L.loc[(L.index>=L.index[-7])&(L.index<=L.index[-1])];b=L.index[-1];N=b+B.Timedelta(days=-6);c=N+B.Timedelta(days=30);s=B.date_range(start=N,end=c);r=Q.loc[s,H];AE=L[H].iloc[-1]
AT()
Aa=AM.datetime.now()
AG=Aa.strftime(AP)

@login_required(login_url='/admin/login/')
def Ab(request):
	T='%B %d, %Y';d=r.iloc[7];g=L[H].iloc[-2];K=P.index[-1];m=K.replace(year=K.year-1);N=B.Timestamp(m);p=P.loc[N,H];q=N.strftime(T);O=P[H].min()
	def s():L='%{y:.2f}';F=P[[H]];F=F.loc[(F.index>=F.index[-7])&(F.index<=F.index[-1])];N=F.index[-1];J=N+B.Timedelta(days=-6);O=J+B.Timedelta(days=7);K=B.date_range(start=J,end=O);R=Q.loc[K,H];S={Z:V,a:E};T=G.Scatter(x=F.index,y=F[H],mode=b,marker=A(color=z,size=5),line=A(width=1.5),name=A0,hovertemplate=L);U=G.Scatter(x=K,y=R,mode=b,marker=A(color=u,size=5),line=A(width=1.5),name=A1,hovertemplate=L);I=G.Figure();I.add_trace(T);I.add_trace(U);I.update_layout(xaxis=A(title=C,titlefont=A(size=14,color=D,family=X),tickformat=A2,tickangle=0,tickfont=A(size=11,color=D,family=X)),yaxis=A(title=AV,titlefont=A(size=15,color=D,family=X),tickfont=A(size=11,color=D,family=X)),margin=A(t=0,l=65,b=70,r=10),plot_bgcolor=M,paper_bgcolor=M,font=A(family=X,size=14,color=D),legend=A(orientation=Y,yanchor=n,y=1.08,xanchor=A3,x=.6),hovermode=A4,hoverlabel=A(bgcolor=A5,font=A(size=15,family=X,color=D)),width=550,height=450,modebar_remove=[h,i,j,k,AJ]);I.update_xaxes(showgrid=E,gridwidth=.5,gridcolor=AK,showspikes=E,spikecolor=D,spikethickness=.7,spikedash=A6);I.update_yaxes(showgrid=E,gridwidth=.5,gridcolor=AK);W=c.to_html(I,config=S);return W
	t=s()
	def w(csv_file):
		A=float('inf');B=v
		with J(csv_file,S,newline='')as G:
			I=csv.DictReader(G)
			for D in I:
				E=D[H]
				if E:
					try:
						F=float(E)
						if F<A:A=F;B=D[C]
					except ValueError:pass
			return A,B
	x=AH;O,F=w(x);A7=AM.datetime.strptime(F,U).date();R=B.Timestamp(A7);AF=P.loc[R,H];F=R.strftime(T)
	def A8():H=B.read_csv(AQ);H[C]=B.to_datetime(H[C],format=U);O=H[C].iloc[-1];L=B.to_datetime(O,format=U);P=L.month;Q=L.year;R=1;N=AN(year=Q,month=P,day=R);S=N.strftime('%Y');T=N.strftime('%B');X=f"{T} {S}";F=H.tail(6);F.set_index(W,inplace=E);F.sort_index(inplace=E);F[l]=F[I]-F[o];J=f(y(e(F)));K=G.Figure(data=[G.Bar(y=J,x=F[I],orientation=Y,name=I,base=0),G.Bar(y=J,x=-F[l],orientation=Y,name='NRWV',base=0)]);K.update_layout(barmode='stack',plot_bgcolor=M,paper_bgcolor=M,font=A(family='Arial, sans-serif',size=14,color=D),title=A(text=X,font=A(size=20,color=D),x=.48,xanchor=AR),xaxis=A(title=A(text=I,font=A(size=16,color=D)),tickfont=A(size=12,color=D)),yaxis=A(title=A(text=W,font=A(size=16,color=D)),tickfont=A(size=12,color=D)));K.update_yaxes(ticktext=F.index,tickvals=J);b={Z:V,a:E,'modeBarButtonsToRemove':[h,i,j,k]};c=K.to_html(config=b);d='\n            <style>\n            .modebar {\n                left: 0;  \n                top: 70px; \n            }\n            </style>\n            ';g=d+c;return g
	A9=A8();AA=Q.index[13];AB=Q.index[14];AC=Q.index[15];return AD(request,'Dashboard.html',{'room_name':'broadcast','Tomorrow':d,'Today':AE,'Yesterday':g,'last_year_today':p,'date_last_year':q,'min_water_level':O,'min_water_level_date':F,C:AG,'date_today':AA,'date_yest':AB,'date_tom':AC,'plot':t,'water_alloc_plot':A9})


@login_required(login_url='/admin/login/')
def Ac(request):
	Ab='forecast_drawdown';Aa='waterlvl_plot.html';A8='drawdown_plot.html';A7='rainfall_plot.html';o='rgba(255, 255, 255, 0.3)';P=request
	def A9():
		P='%{y:.2f} m';AT();R=L.index[-1];N=R+B.Timedelta(days=-6);S=N+B.Timedelta(days=470);I=B.date_range(start=N,end=S);T=Q.loc[I,H];f=L[H].iloc[-1];U={Z:V,a:E};W=G.Scatter(x=L.index,y=L[H],mode=b,marker=A(color=z,size=5),line=A(width=1.5),name=A0,hovertemplate=P);d=G.Scatter(x=I,y=T,mode=b,marker=A(color=u,size=5),line=A(width=1.5),name=A1,hovertemplate=P);F=G.Figure();F.add_trace(W);F.add_trace(d);F.update_layout(xaxis=A(title=C,titlefont=A(size=14,color=D),tickformat=A2,tickangle=0,tickfont=A(size=12,color=D),range=[I[0],I[30]]),yaxis=A(title=AV,titlefont=A(size=15,color=D),tickfont=A(size=12,color=D)),margin=A(t=10,l=100,b=10,r=10),plot_bgcolor=M,paper_bgcolor=M,font=A(family=p,size=14,color=D),legend=A(orientation=Y,yanchor=n,y=1.08,xanchor=A3,x=0),hovermode=A4,hoverlabel=A(bgcolor=A5,font=A(size=15,family=X,color=D)),width=990,height=600,modebar_remove=[h,i,j,k]);F.update_xaxes(showgrid=E,gridwidth=.5,gridcolor=o,showspikes=E,spikecolor=D,spikethickness=.7,spikedash=A6);F.update_yaxes(showgrid=E,gridwidth=.5,gridcolor=o);O=c.to_html(F,config=U)
		with J(Aa,q,encoding=K)as e:e.write(O)
		return O
	Ac=Q.index[14];AA=Q[H].iloc[14];AA=T(AA,2);r=A9();AW=x.predict(AF);N=x.predict(t);AW=m.inverse_transform(F.concatenate((AW,AF[:,-1,1:]),axis=1))[:,0];N=m.inverse_transform(F.concatenate((N,t[:,-1,1:]),axis=1))[:,0];s=m.inverse_transform(F.concatenate((AU.reshape(-1,1),t[:,-1,1:]),axis=1))[:,0];N=F.array(N);s=F.array(s);Ad=R(s-N);Ae=Ad/(R(s)+R(N));AB=100*F.mean(Ae);AB=T(AB,2);Af=R(Q[H].iloc[14]-L[H].iloc[-1]);Ag=R(L[H].iloc[-1])+R(Q[H].iloc[14]);Ah=Af/Ag;AJ=100*Ah;AJ=T(AJ,2)
	def AX():
		AJ='%{y:.2f} mm';AH='WIND_DIRECTION';AG='WIND_SPEED';H='RAINFALL';m=AC('Model_rainfall.keras');O=B.read_csv('rainfall_data.csv');O[H]=B.to_numeric(O[H],errors=AI);g=d(e(O)*.8);P=O.iloc[:g];Al=O.iloc[g:];AK=P['TMAX'].mean();AL=P['TMIN'].mean();AM=P['TMEAN'].mean();AN=P[AG].mean();AO=P[AH].mean();AP=P['RH'].mean();L=O.fillna(value={H:0,'TMAX':AK,'TMIN':AL,'TMEAN':AM,AG:AN,AH:AO,'RH':AP}).copy();L[C]=B.to_datetime(L[['YEAR','MONTH','DAY']],format=U);L.set_index(C,inplace=E);L.drop(columns=['YEAR','DAY','MONTH'],inplace=E);W=AS();x=W.fit_transform(L)
		def A8(data,seq_length):
			B=seq_length;A=data;D=[];E=[]
			for C in y(e(A)-B):D.append(A[C:C+B,:]);E.append(A[C+B,0])
			return F.array(D),F.array(E)
		A9=12;r,Am=A8(x[:g],A9);s,AQ=A8(x[g:],A9);AR=f(L.index);AT=10;AA=300;AU=B.date_range(f(AR)[-AT],periods=AA,freq='d').tolist();AV=m.predict(r[-AA:]);AW=F.repeat(AV,L.shape[1],axis=-1);AX=W.inverse_transform(AW)[:,0];N=[]
		for AY in AU:N.append(AY.date())
		Q=B.DataFrame({C:F.array(N),H:AX});Q.set_index(C,inplace=E);I=L[[H]];I=I.loc[(I.index>=I.index[-10])&(I.index<=I.index[-1])];AZ=I.index[-1];AB=AZ+B.Timedelta(days=-6);Aa=AB+B.Timedelta(days=295);N=B.date_range(start=AB,end=Aa);Ab=Q.loc[N,H];Ac={Z:V,a:E};Ad=G.Scatter(x=I.index,y=I[H],mode=b,marker=A(color=z,size=5),line=A(width=1.5),name=A0,hovertemplate=AJ);Ae=G.Scatter(x=N,y=Ab,mode=b,marker=A(color=u,size=5),line=A(width=1.5),name=A1,hovertemplate=AJ);S=G.Figure();S.add_trace(Ad);S.add_trace(Ae);S.update_layout(xaxis=A(title=C,titlefont=A(size=14,color=D),tickformat=A2,tickangle=0,tickfont=A(size=12,color=D),range=[N[0],N[30]]),yaxis=A(title='Rainfall (mm)',titlefont=A(size=15,color=D),tickfont=A(size=12,color=D)),margin=A(t=0,l=100,b=10,r=10),plot_bgcolor=M,paper_bgcolor=M,font=A(family=p,size=14,color=D),legend=A(orientation=Y,yanchor=n,y=1.08,xanchor=A3,x=0),hovermode=A4,hoverlabel=A(bgcolor=A5,font=A(size=15,family=X,color=D)),width=990,height=600,modebar_remove=[h,i,j,k]);S.update_xaxes(showgrid=E,gridwidth=.5,gridcolor=o,showspikes=E,spikecolor=D,spikethickness=.7,spikedash=A6);S.update_yaxes(showgrid=E,gridwidth=.5,gridcolor=o);AD=c.to_html(S,config=Ac)
		with J(A7,q,encoding=K)as Af:Af.write(AD)
		AE=m.predict(r);l=m.predict(s);AE=W.inverse_transform(F.concatenate((AE,r[:,-1,1:]),axis=1))[:,0];l=W.inverse_transform(F.concatenate((l,s[:,-1,1:]),axis=1))[:,0];AF=W.inverse_transform(F.concatenate((AQ.reshape(-1,1),s[:,-1,1:]),axis=1))[:,0];l=F.array(l);AF=F.array(AF);Ag=R(Q[H].iloc[9]-I[H].iloc[-1]);Ah=R(I[H].iloc[-1])+R(Q[H].iloc[9]);Ai=Ag/Ah;t=100*Ai;t=T(t,2);v=Q[H].iloc[9];v=T(v,2);Aj=I[H].iloc[-1];Ak=Q.index[9];return w,t,v,Aj,Ak,AD
	AL=0;AM=0;w=.0;AN=.0
	def AY():
		AJ='%{y:.2f} cu m';t=AC('Model_drawdown.h5');L=B.read_csv(AH);AL=[C,O,g,H];L=L[AL];L[g]=B.to_numeric(L[g],errors=AI);m=d(e(L)*.8);x=L.iloc[:m];Aj=L.iloc[m:];AM=x[H].mean();AN=x[O].mean();N=L.fillna(value={O:AN,g:0,H:AM}).copy();N[C]=B.to_datetime(N[C],format=U);N.set_index(C,inplace=E);W=AS();A7=W.fit_transform(N)
		def A9(data,seq_length):
			B=seq_length;A=data;D=[];E=[]
			for C in y(e(A)-B):D.append(A[C:C+B,:]);E.append(A[C+B,0])
			return F.array(D),F.array(E)
		AA=10;AB,Ak=A9(A7[:m],AA);r,AO=A9(A7[m:],AA);AP=f(N.index);AQ=89;AD=221;AR=B.date_range(f(AP)[-AQ],periods=AD,freq='d').tolist();AT=t.predict(r[-AD:]);AU=F.repeat(AT,N.shape[1],axis=-1);AV=W.inverse_transform(AU)[:,0];P=[]
		for AW in AR:P.append(AW.date())
		l=B.DataFrame({C:F.array(P),O:AV});l.set_index(C,inplace=E);I=N[[O]];AX=I[O].iloc[-1];I=I.loc[(I.index>=I.index[-7])&(I.index<=I.index[-1])];AY=I.index[-1];AE=AY+B.Timedelta(days=-6);AZ=AE+B.Timedelta(days=132);P=B.date_range(start=AE,end=AZ);Aa=l.loc[P,O];Ab={Z:V,a:E};Ac=G.Scatter(x=I.index,y=I[O],mode=b,marker=A(color=z,size=5),line=A(width=1.5),name=A0,hovertemplate=AJ);Ad=G.Scatter(x=P,y=Aa,mode=b,marker=A(color=u,size=5),line=A(width=1.5),name=A1,hovertemplate=AJ);Q=G.Figure();Q.add_trace(Ac);Q.add_trace(Ad);Q.update_layout(xaxis=A(title=C,titlefont=A(size=14,color=D),tickformat=A2,tickangle=0,tickfont=A(size=12,color=D),range=[P[0],P[30]]),yaxis=A(title='Drawdown (cu m)',titlefont=A(size=15,color=D),tickfont=A(size=12,color=D)),margin=A(t=0,l=100,b=10,r=10),plot_bgcolor=M,paper_bgcolor=M,font=A(family=p,size=14,color=D),legend=A(orientation=Y,yanchor=n,y=1.08,xanchor=A3,x=0),hovermode=A4,hoverlabel=A(bgcolor=A5,font=A(size=15,family=X,color=D)),width=990,height=600,modebar_remove=[h,i,j,k]);Q.update_xaxes(showgrid=E,gridwidth=.5,gridcolor=o,showspikes=E,spikecolor=D,spikethickness=.7,spikedash=A6);Q.update_yaxes(showgrid=E,gridwidth=.5,gridcolor=AK);AF=c.to_html(Q,config=Ab)
		with J(A8,q,encoding=K)as Ae:Ae.write(AF)
		AG=t.predict(AB);S=t.predict(r);AG=W.inverse_transform(F.concatenate((AG,AB[:,-1,1:]),axis=1))[:,0];S=W.inverse_transform(F.concatenate((S,r[:,-1,1:]),axis=1))[:,0];s=W.inverse_transform(F.concatenate((AO.reshape(-1,1),r[:,-1,1:]),axis=1))[:,0];S=F.array(S);s=F.array(s);v=F.mean(F.abs(S-s)/F.abs(S+s)/2)*100;v=T(v,2);Af=R(l[O].iloc[88]-I[O].iloc[-1]);Ag=(R(I[O].iloc[-1])+R(l[O].iloc[88]))/2;Ah=Af/Ag;w=100*Ah;w=T(w,2);Ai=l[O].iloc[88];return v,w,Ai,AX,AF
	AO=0;AP=0;AQ=.0;AR=.0;Ai=P.GET.get('forecast_all',v);Aj=P.GET.get('forecast_waterlvl',v);Ak=P.GET.get('forecast_rainfall',v);Al=P.GET.get(Ab,v)
	if Ai:r=A9();w,AN,AL,AM,AZ,W=AX();AQ,AR,AO,AP,l=AY()
	elif Aj:
		r=A9()
		with J(A7,S,encoding=K)as I:W=I.read()
		with J(A8,S,encoding=K)as I:l=I.read()
	elif Ak:
		w,AN,AL,AM,AZ,W=AX()
		with J(A8,S,encoding=K)as I:l=I.read()
	elif Al:
		AQ,AR,AO,AP,l=AY()
		with J(A7,S,encoding=K)as I:W=I.read()
	else:
		with J(Aa,S,encoding=K)as I:r=I.read()
		with J(A7,S,encoding=K)as I:W=I.read()
		with J(A8,S,encoding=K)as I:l=I.read()
	with J('water_level_test_set.html',S,encoding=K)as I:Am=I.read()
	with J('rainfall_test_set.html',S,encoding=K)as I:An=I.read()
	with J('drawdown_test_set.html',S,encoding=K)as I:Ao=I.read()
	return AD(P,'Forecast.html',{C:AG,'actual':AE,'forecasted':AA,'forecasted_date':Ac,'fore_smape':AB,'act_smape':AJ,Ab:AO,'actual_drawdown':AP,'fore_drawdown_smape':AQ,'act_drawdown_smape':AR,'actual_rain':AM,'forecast_rain':AL,'fore_rain_smape':w,'act_rain_smape':AN,'water_plot':r,'rain_plot':W,'drawdown_interact_plot':l,'water_level_test_set':Am,'rainfall_test_set':An,'drawdown_test_set':Ao})

@login_required(login_url='/admin/login/')
def Ad(request):
	AS='bar_chart.html';O=request;global N;H=B.read_csv(AQ);H[C]=B.to_datetime(H[C],format=U);AT=H[C].iloc[-1];AM=B.to_datetime(AT,format=U);AO=AM.month;AP=AM.year;P=1;AU=1
	if O.method=='POST':AO=d(O.POST['month']);AP=d(O.POST['year']);P=d(O.POST['graph'])
	T=AN(year=AP,month=AO,day=AU);AV=T.strftime('%Y');AY=T.strftime('%B');X=f"{AY} {AV}";AZ=X;Aa=T.strftime(U);N=H[H[C].dt.strftime(U)==Aa];N.set_index(W,inplace=E);N.sort_index(inplace=E);Q=N.index;N[l]=N[I]-N[o]
	def Ab():
		B=G.Figure();F={Z:V,a:E};B.add_trace(G.Bar(x=Q,y=N[I],name=I,marker_color='blue'));B.add_trace(G.Bar(x=Q,y=N[l],name='NRWV',marker_color='skyblue'));B.update_layout(title='BAR CHART',xaxis_title=W,yaxis_title='Volume',barmode='stack',plot_bgcolor=M,paper_bgcolor=M,font=A(family=p,size=14,color=D),width=1000,height=600,legend=A(orientation=Y,yanchor='bottom',y=1.02,xanchor='right',x=1),modebar_remove=[h,'pan',i,'pan2d',j,k,AJ]);C=c.to_html(B,config=F)
		with J(AS,q,encoding=K)as H:H.write(C)
		return C
	def Ac():
		O='inside';L='label+percent';C=u,'cyan','brown','grey','indigo','beige';P={Z:V,a:E}
		def S(pct,allvalues):A=d(pct/1e2*F.sum(allvalues));return'{:.1f}%\n({:d} mld)'.format(pct,A)
		B=G.Figure();B.add_trace(G.Pie(labels=Q,values=N[l],textinfo=L,textposition=O,marker=A(colors=C),sort=V,domain={'x':[.55,1],'y':[.05,.95]},title='Non-Revenue Water Volume'));B.add_trace(G.Pie(labels=Q,values=N[I],textinfo=L,textposition=O,marker=A(colors=C),sort=V,domain={'x':[0,.45],'y':[.05,.95]},title=I));B.update_layout(title='PIE CHART',showlegend=E,plot_bgcolor=M,paper_bgcolor=M,font=A(family=p,size=14,color=D),legend=A(orientation=Y,yanchor=n,y=0,xanchor=AR,x=.5),width=950,height=600,margin=A(t=50,b=10,l=10,r=10));H=c.to_html(B,config=P)
		with J('pie_chart.html',q,encoding=K)as R:R.write(H)
		return H
	def Ad():
		T='rgba(200, 200, 200, 0.5)';F='lines+markers';L=H[H[W]==A7];N=H[H[W]==A8];O=H[H[W]==A9];P=H[H[W]==AA];Q=H[H[W]==AB];R=H[H[W]==AL];B=G.Figure();U={Z:V,a:E};B.add_trace(G.Scatter(x=L[C],y=L[I],mode=F,name=A7));B.add_trace(G.Scatter(x=N[C],y=N[I],mode=F,name=A8));B.add_trace(G.Scatter(x=O[C],y=O[I],mode=F,name=A9));B.add_trace(G.Scatter(x=P[C],y=P[I],mode=F,name=AA));B.add_trace(G.Scatter(x=Q[C],y=Q[I],mode=F,name=AB));B.add_trace(G.Scatter(x=R[C],y=R[I],mode=F,name='Up Katipunan'));B.update_layout(title='LINE CHART',xaxis_title=C,yaxis_title=I,plot_bgcolor=M,paper_bgcolor=M,xaxis=A(showgrid=E,gridcolor=T,gridwidth=1),yaxis=A(showgrid=E,gridcolor=T,gridwidth=1),font=A(family=p,size=14,color=D),legend=A(orientation=Y,yanchor=n,y=1.08,xanchor='right',x=1),width=980,height=600,modebar_remove=[h,i,j,k,AJ]);S=c.to_html(B,config=U)
		with J('line_chart.html',q,encoding=K)as X:X.write(S)
		return S
	if P==2:R=Ab()
	elif P==3:R=Ac()
	elif P==4:R=Ad();X='Monthly'
	else:
		with J(AS,S,encoding=K)as Ae:R=Ae.read()
	b=N[I].sum();e=N[l].sum();Af=e/b*100;Ag=b-e
	def L(location):A=N.loc[location];C=A[I];D=A[o];B=A[I]-A[o];E=A[I]-B;return C,D,B,E
	f=0;g=0;m=0;r=0;f,g,m,r=L(A7);s=0;t=0;v=0;w=0;s,t,v,w=L(A8);x=0;y=0;z=0;A0=0;x,y,z,A0=L(A9);A1=0;A2=0;A3=0;A4=0;A1,A2,A3,A4=L(AA);A5=0;A6=0;AC=0;AE=0;A5,A6,AC,AE=L(AB);AF=0;AH=0;AI=0;AK=0;AF,AH,AI,AK=L(AL);Ah=f+s+x+A1+A5+AF;Ai=g+t+y+A2+A6+AH;Aj=m+v+z+A3+AC+AI;Ak=r+w+A0+A4+AE+AK;return AD(O,'Business-zones.html',{C:AG,'supply':b,'total_supply':Ag,'total_nrwv':e,AW:Af,'display_date':X,'month_date':AZ,'chart':R,'araneta_sv':f,'araneta_bill':g,'araneta_nrwv':m,'araneta_ws':r,'elli_sv':s,'elli_bill':t,'elli_nrwv':v,'elli_ws':w,'sj_sv':x,'sj_bill':y,'sj_nrwv':z,'sj_ws':A0,'ts_sv':A1,'ts_bill':A2,'ts_nrwv':A3,'ts_ws':A4,'timog_sv':A5,'timog_bill':A6,'timog_nrwv':AC,'timog_ws':AE,'up_sv':AF,'up_bill':AH,'up_nrwv':AI,'up_ws':AK,'supply_volume':Ah,'bill_volume':Ai,'nrw_volume':Aj,AX:Ak})

@login_required(login_url='/admin/login/')
def Ae(request):
	K='error';G=request
	if G.headers.get('X-Requested-With')=='XMLHttpRequest':
		if G.method=='GET':
			J=B.DataFrame(N);A=G.GET.get('location_name')
			if A in J.index:
				F=J.loc[A];D=F[I]-F[o];L=F[o];E=F[I];H=D/E*100;M=E-D;D=T(D,2);E=T(E,2);H=T(H,2);C=''
				if A==A8:C='img/bz-map(elliptical).png'
				elif A==AA:C='img/bz-map(tsora).png'
				elif A==AB:C='img/bz-map(timog).png'
				elif A==AL:C='img/bz-map(up).png'
				elif A==A7:C='img/bz-map(araneta).png'
				elif A==A9:C='img/bz-map(sjuan).png'
				O={l:D,'sv':E,'bv':L,AX:M,AW:H,'location':A,'img_src':static(C)if C else''};return w(O)
			else:return w({K:'Location not found'},status=404)
	return w({K:'Invalid request'},status=400)
def Af(request):A=AO.localtime(AO.now());B=A.strftime(AP);return w({'current_datetime':B})