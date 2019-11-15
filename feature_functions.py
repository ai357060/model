import numpy as np
import pandas as pd
from scipy import stats
import scipy.optimize
from scipy.optimize  import OptimizeWarning
import warnings
import math
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from datetime import datetime
from arch import arch_model
import datetime as dt
from statsmodels.tsa.arima_model import ARIMA
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error

class holder:
    1

slopeperiods = [3,4,5,10,20,30]    

def loaddata_master(datafile='usdjpy_d.csv'):
    df = pd.read_csv('C:/Users/pioo/Desktop/Temp/jpynotebooks/WinMachine/Data/'+datafile,sep=';')
    try:
        df.fulldate=pd.to_datetime(df.fulldate,format='%Y-%m-%d')
    except:
        df.fulldate=pd.to_datetime(df.fulldate,format='%Y.%m.%d')    

    return df

def loaddata(datafile='usdjpy_d.csv'):
    df = pd.read_csv('C:/Users/pioo/Desktop/Temp/jpynotebooks/WinMachine/Data/'+datafile)
    #df.columns = ['date','open','high','low','close']
    try:
        df.date=pd.to_datetime(df.date,format='%Y-%m-%d')
    except:
        df.date=pd.to_datetime(df.date,format='%Y.%m.%d')    

        
    df = df.set_index(df.date)
    df['year'] = pd.DatetimeIndex(df['date']).year
    df['month'] = pd.DatetimeIndex(df['date']).month
    df['day'] = pd.DatetimeIndex(df['date']).day 
    df = df.drop(['date'],1)
    return df
    
def loaddata_nodateindex(datafile='usdjpy_d.csv'):
    df = pd.read_csv('../Data/'+datafile)
    #df.columns = ['date','open','high','low','close']
    try:
        df.date=pd.to_datetime(df.date,format='%Y-%m-%d')
    except:
        #df.date=pd.to_datetime(df.date,format='%Y.%m.%d')    
        df.date=pd.to_datetime(df.date,format='%d.%m.%Y %H:%M:%S.%f')    

    df=df[df.volume!=0]
    
    df['year'] = pd.DatetimeIndex(df['date']).year
    df['month'] = pd.DatetimeIndex(df['date']).month
    df['day'] = pd.DatetimeIndex(df['date']).day 
    df = df.rename(index=str, columns={"date": "fulldate"})
    df.index.names = ['date']
    
    return df
    
#Momentum
def momentum(prices,periods):
    '''
    :param prices; dataframe of OHLC currency data
    :param periods; list of periods for which to compute the function [3,5,10,...]
    :return; momentum indicator
    '''    

    results = holder()
    dict = {}
    for i in range(0, len(periods)):
        resdf = pd.DataFrame(index=prices.index)
        resdf['open'] = prices.open.diff(periods=periods[i])
        resdf['close'] = prices.close.diff(periods=periods[i])
        resdf['open_div'] = 100 * prices.open / prices.open.shift(periods=periods[i])
        resdf['close_div'] =100 *  prices.close / prices.close.shift(periods=periods[i])
        dict[periods[i]] = resdf
    
    results.df = dict
    return results
  
def stochastic(prices, periods,InpSlowing=3):
    '''
    :param prices; dataframe of OHLC currency data
    :param periods; list of periods for which to compute the function [3,5,10,...]
    :return; stochastic indicator
    ''' 
    results = holder()
    
    dict = {}
    for i in range(0,len(periods)):
        stoch = pd.DataFrame(index=prices.index)
        stoch['C'] = prices.close
        stoch['H'] = prices.high.rolling(periods[i]).max()
        stoch['L'] = prices.low.rolling(periods[i]).min()
        stoch['CL'] = stoch['C'] - stoch['L']
        stoch['HL'] = stoch['H'] - stoch['L']
        stoch['rCL'] = stoch['CL'].rolling(InpSlowing).sum()
        stoch['rHL'] =stoch['HL'].rolling(InpSlowing).sum()
        stoch['K'] = 0
        stoch.loc[stoch['HL']!=0,'K'] = 100 * stoch['rCL']/stoch['rHL']
            
        stoch['D'] = stoch.K.rolling(3).mean()
        stoch['HistKD'] = stoch['K'] - stoch['D']

        ## slopes
        slopecolumnlist = ['K','HistKD']
        sl = slopecolumns(stoch,slopeperiods,slopecolumnlist)
        for kk in range(0,len(slopecolumnlist)):
            for ss in range(0,len(slopeperiods)):
                sscolID = slopecolumnlist[kk]+'_slope' + str(slopeperiods[ss])
                stoch[sscolID]=sl.df[slopeperiods[ss]][sscolID];
                
        dict[periods[i]] = stoch
    results.df = dict
    return results

def williams(prices, periods):
    '''
    :param prices; dataframe of OHLC currency data
    :param periods; list of periods for which to compute the function [3,5,10,...]
    :return; williams osc
    ''' 
    results = holder()
    
    dict = {}
    for i in range(0,len(periods)):
        w = pd.DataFrame(columns = ['date','C','H','L','R'])
        for j in range(periods[i]-1,len(prices)):
            C = prices.close.iloc[j]
            H = prices.high.iloc[j-periods[i]+1:j+1].max()
            L = prices.low.iloc[j-periods[i]+1:j+1].min()
            if H==L:
                R=0
            else:
                R=-100*(H-C)/(H-L)
    
            w = w.append({'date':prices.index[j],'C':C,'H':H,'L':L,'R':R},ignore_index=True)
        w = w.set_index(w.date)
        w = w[['C','H','L','R']]
        dict[periods[i]] = w
    results.df = dict
    return results

#Price rate of change
def proc(prices,periods):
    '''
    :param prices; dataframe of OHLC currency data
    :param periods; list of periods for which to compute the function [3,5,10,...]
    :return; price rate of change
    '''    

    results = holder()
    dict = {}
    for i in range(0, len(periods)):
        close = pd.DataFrame()
        close['close_pct'] = prices.close.pct_change(periods=periods[i])
        close['tp3'] = prices[['close','high','low']].mean(axis = 1) 
        close['tp3logdiff'] = np.log(close['tp3']).diff(periods=periods[i]-1) 
        close = close[['close_pct','tp3logdiff']]
        dict[periods[i]] = close
    
    results.df = dict
    return results

#Williams Accumulation Distribution Function
def wadl(prices,periods):
    '''
    :param prices; dataframe of OHLC currency data
    :param periods; list of periods for which to compute the function [3,5,10,...]
    :return; williams accumulation distribution lines for each period
    '''    
    results = holder()
    dict = {}
    for i in range(0,len(periods)):
        WAD = pd.DataFrame(columns = ['date','wad'])
        for j in range(periods[i]-1,len(prices)):
            t = periods[i] - 1
            TRH = np.array([prices.high[j],prices.close[j-t]]).max()
            TRL = np.array([prices.low[j],prices.close[j-t]]).min()
            if prices.close[j] > prices.close[j-t]:
                PM = prices.close[j] - TRL
            elif prices.close[j] < prices.close[j-t]:
                PM = prices.close[j] - TRH
            else:
                PM = 0
            #AD = PM * prices.volume[j]
            AD = PM
            WAD = WAD.append({'date':prices.index[j],'wad':AD},ignore_index=True)
        WAD = WAD.set_index(WAD.date)
        WAD = WAD[['wad']]
        WAD['wad'] = WAD.wad.cumsum()
        ## slopes
        slopecolumnlist = ['wad']
        sl = slopecolumns(WAD,slopeperiods,slopecolumnlist)
        for kk in range(0,len(slopecolumnlist)):
            for ss in range(0,len(slopeperiods)):
                sscolID = slopecolumnlist[kk]+'_slope' + str(slopeperiods[ss])
                WAD[sscolID]=sl.df[slopeperiods[ss]][sscolID];

        dict[periods[i]] = WAD
    results.df = dict
    return results

def adosc(prices, periods):
    '''
    :param prices; dataframe of OHLC currency data
    :param periods; list of periods for which to compute the function [3,5,10,...]
    :return; adosc indicator
    ''' 
    results = holder()
    
    dict = {}
    for i in range(0,len(periods)):
        adosc = pd.DataFrame(columns = ['date','C','H','L','ADL','ADL3','ADL10','Chaikin'])
        for j in range(periods[i]-1,len(prices)):
            C = prices.close.iloc[j]
            H = prices.high.iloc[j-periods[i]+1:j+1].max()
            L = prices.low.iloc[j-periods[i]+1:j+1].min()
            V = prices.volume.iloc[j-periods[i]+1:j+1].sum()
            if H==L:
                MFV=0
            else:
                MFV=((C-L)-(H-C))/(H-L)
            
            ADL = MFV * V
    
            adosc = adosc.append({'date':prices.index[j],'C':C,'H':H,'L':L,'ADL':ADL},ignore_index=True)
        adosc = adosc.set_index(adosc.date)
        adosc['ADL']   = adosc.ADL.cumsum()
        adosc['ADL3']  = adosc.ADL.rolling(3).mean()
        adosc['ADL10'] = adosc.ADL.rolling(10).mean()
        adosc['Chaikin'] = adosc.ADL3 - adosc.ADL10
        adosc = adosc[['C','H','L','ADL','ADL3','ADL10','Chaikin']]
        
        ## slopes
        slopecolumnlist = ['ADL','Chaikin']
        sl = slopecolumns(adosc,slopeperiods,slopecolumnlist)
        for kk in range(0,len(slopecolumnlist)):
            for ss in range(0,len(slopeperiods)):
                sscolID = slopecolumnlist[kk]+'_slope' + str(slopeperiods[ss])
                adosc[sscolID]=sl.df[slopeperiods[ss]][sscolID];
                
        dict[periods[i]] = adosc
    results.df = dict
    return results

def macd(prices,periods):
    '''
    :param prices; dataframe of OHLC currency data
    :param periods; ema1, ema2, signal period
    :return; macd
    '''
    results = holder()
    
    EMA1 = prices.close.ewm(span=periods[0]).mean()
    EMA2 = prices.close.ewm(span=periods[1]).mean()
    macddf = pd.DataFrame(index=prices.index)
    macddf['MACD'] = EMA1-EMA2
    macddf['SigMACD'] = macddf.MACD.rolling(periods[2]).mean()
    macddf['HistMACD'] = macddf['MACD'] - macddf['SigMACD']
    ## slopes
    slopecolumnlist = ['MACD','HistMACD']
    sl = slopecolumns(macddf,slopeperiods,slopecolumnlist)
    for kk in range(0,len(slopecolumnlist)):
        for ss in range(0,len(slopeperiods)):
            sscolID = slopecolumnlist[kk]+'_slope' + str(slopeperiods[ss])
            macddf[sscolID]=sl.df[slopeperiods[ss]][sscolID];    
    
    dict = {}
    dict[periods[0]] = macddf
    results.df = dict
    return results
    
def cci(prices, periods):
    '''
    :param prices; dataframe of OHLC currency data
    :param periods; 
    :return; cci
    '''

    results = holder()
    dict = {}
    for i in range(0,len(periods)):
        ccidf = pd.DataFrame(index = prices.index)
        ccidf['tp']= (prices.high+prices.low+prices.close)/3
        ccidf['tpmean'] = ccidf.tp.rolling(periods[i]).mean()
        ccidf = ccidf.drop(ccidf[ccidf.tpmean.isna()].index,axis=0)
        ccidf['tpdiff']  = np.NaN
        ccidf['cci'] = 0
        for j in range(periods[i]-1,len(ccidf)):
            absdiff = 0
            for k in range(0,periods[i]):
                absdiff += abs(ccidf.tp.iloc[j-k] - ccidf.tpmean.iloc[j])
            ccidf.iloc[j:j+1,[2]] = absdiff * 0.015 / periods[i]
        ccidf.cci = (ccidf.tp - ccidf.tpmean)/ccidf.tpdiff
        ## slopes
        slopecolumnlist = ['cci']
        sl = slopecolumns(ccidf,slopeperiods,slopecolumnlist)
        for kk in range(0,len(slopecolumnlist)):
            for ss in range(0,len(slopeperiods)):
                sscolID = slopecolumnlist[kk]+'_slope' + str(slopeperiods[ss])
                ccidf[sscolID]=sl.df[slopeperiods[ss]][sscolID];         

        dict[periods[i]] = ccidf
    results.df = dict
    return results


def bollinger(prices, periods):
    '''
    :param prices; dataframe of OHLC currency data
    :param periods; 
    :return; bb
    '''

    results = holder()
    dict = {}
    for i in range(0,len(periods)):
        bbdf = pd.DataFrame()
        bbdf['mid'] = prices.close.rolling(periods[i]).mean()
        bbdf['upper']  = bbdf.mid + 2* prices.close.rolling(periods[i]).std()
        bbdf['lower']  = bbdf.mid - 2* prices.close.rolling(periods[i]).std()
        bbdf['Histmid'] = prices.close - bbdf.mid
        bbdf['Histupper']  = prices.close - bbdf.upper
        bbdf['Histlower']  = prices.close - bbdf.lower
        
        ## slopes
        slopecolumnlist = ['Histmid','Histupper','Histlower']
        sl = slopecolumns(bbdf,slopeperiods,slopecolumnlist)
        for kk in range(0,len(slopecolumnlist)):
            for ss in range(0,len(slopeperiods)):
                sscolID = slopecolumnlist[kk]+'_slope' + str(slopeperiods[ss])
                bbdf[sscolID]=sl.df[slopeperiods[ss]][sscolID];         
                
        dict[periods[i]] = bbdf
        
        
    results.df = dict
    return results

#Heiken Ashi Candless    
def heikenashi(prices,periods):
    '''
    :param prices - dataframe of OHLC
    :param periods - used to rolling resample
    :return - heiken OHLC candles
    '''
    results = holder()
    dict={}
    for i in range(0,len(periods)):
        res = OHLCresample(prices,periods[i])
        df = res.df
        resdf = pd.DataFrame(columns = ['HAopen','HAhigh','HAlow','HAclose'])
        resdf['HAclose'] = df[['RSopen','RShigh','RSlow','RSclose']].sum(axis=1)/4
        resdf['HAopen'] = resdf.HAclose.copy()
        resdf['HAhigh'] = resdf.HAclose.copy()
        resdf['HAlow']  = resdf.HAclose.copy()
        for j in range(periods[i],len(prices)):
            resdf.HAopen.iloc[j] = (resdf.HAopen.iloc[j - periods[i]] + resdf.HAclose.iloc[j - periods[i]])/2
            resdf.HAhigh.iloc[j] = np.array([df.RShigh.iloc[j],resdf.HAopen.iloc[j],resdf.HAclose.iloc[j]]).max()
            resdf.HAlow.iloc[j]  = np.array([df.RSlow.iloc[j],resdf.HAopen.iloc[j],resdf.HAclose.iloc[j]]).min()
        
        ## slopes
        slopecolumnlist = ['HAopen','HAhigh','HAlow','HAclose']
        sl = slopecolumns(resdf,slopeperiods,slopecolumnlist)
        for kk in range(0,len(slopecolumnlist)):
            for ss in range(0,len(slopeperiods)):
                sscolID = slopecolumnlist[kk]+'_slope' + str(slopeperiods[ss])
                resdf[sscolID]=sl.df[slopeperiods[ss]][sscolID];         
        
        dict[periods[i]] = resdf
    
    results.df=dict
    return results

def OHLCresample(prices, period):
    #rolling version
    #period = 1 means no resample
    results = holder()
    a = pd.DataFrame(columns=['RSopen','RShigh','RSlow','RSclose'])
    a['RSclose'] = prices.close
    a['RSopen']  = prices.open.shift(periods = period-1)
    a['RShigh']  = prices.high.rolling(period).max()
    a['RSlow']   = prices.low.rolling(period).min()

    for j in range(0,period-1):
        a.RSclose[j] = prices.close[j]
        a.RSopen[j] = prices.open[j]
        a.RShigh[j] = prices.high[j]
        a.RSlow[j] = prices.low[j]    
    
    results.df = a
    return results

def heikenashi_old(prices,periods):
    '''
    :param prices - dataframe of OHLC
    :param periods - nie jest brane pod uwagę
    :return - heiken OHLC candles
    '''
    results = holder()
    dict={}
    HAclose=prices[['open','high','low','close']].sum(axis=1)/4
    HAopen=HAclose.copy()
    HAopen.iloc[0]=HAclose.iloc[0]
    HAhigh=HAclose.copy()
    HAlow=HAclose.copy()
    
    for i in range(1,len(prices)):
        HAopen.iloc[i]=(HAopen.iloc[i-1]+HAclose.iloc[i-1])/2
        HAhigh.iloc[i]=np.array([prices.high.iloc[i],HAopen.iloc[i],HAclose.iloc[i]]).max()
        HAlow.iloc[i]=np.array([prices.low.iloc[i],HAopen.iloc[i],HAclose.iloc[i]]).min()
    
    df = pd.concat((HAopen,HAhigh,HAlow,HAclose),axis=1)
    df.columns = ['open','high','low','close']
    dict[periods[0]]=df
    results.df=dict
    return results

def OHLCresample_old(prices, TimeFrame):
    
    results = holder()
    a = pd.DataFrame(columns=['open','high','low','close'])
    a['open'] = prices.open.resample(TimeFrame,label='right',closed='right').first()
    a['close'] = prices.close.resample(TimeFrame,label='right',closed='right').last()
    a['high'] = prices.high.resample(TimeFrame,label='right',closed='right').max()
    a['low'] = prices.low.resample(TimeFrame,label='right',closed='right').min()
    
    results.df = a
    return results



def paverages(prices, periods):
    '''
    :param prices; dataframe of OHLC currency data
    :param periods; 
    :return; avg
    '''

    results = holder()
    dict = {}
    for i in range(0,len(periods)):
        bbdf = pd.DataFrame()
        bbdf['avg_close'] = prices.close.rolling(periods[i]).mean()
        bbdf['avg_open'] = prices.open.rolling(periods[i]).mean()
        bbdf['avg_low'] = prices.low.rolling(periods[i]).mean()
        bbdf['avg_high'] = prices.high.rolling(periods[i]).mean()
        bbdf['tp4']= (prices.high+prices.low+prices.close+prices.open)/4
        bbdf['weightedtp4']= (prices.high+prices.low+prices.close+prices.close)/4
        bbdf['highlow']= (prices.high+prices.low)/2
        bbdf['tp4mean'] = bbdf.weightedtp4.rolling(periods[i]).mean()
        bbdf['highlowmean'] = bbdf.highlow.rolling(periods[i]).mean()
        
        
        ## slopes
        slopecolumnlist = ['avg_close','tp4mean','highlowmean','weightedtp4']
        sl = slopecolumns(bbdf,slopeperiods,slopecolumnlist)
        for kk in range(0,len(slopecolumnlist)):
            for ss in range(0,len(slopeperiods)):
                sscolID = slopecolumnlist[kk]+'_slope' + str(slopeperiods[ss])
                bbdf[sscolID]=sl.df[slopeperiods[ss]][sscolID];         
        
        dict[periods[i]] = bbdf
        
    results.df = dict
    return results

def slopes(prices, periods):
    '''
    :param prices; dataframe of OHLC currency data
    :param periods; list of periods for which to compute the function [3,5,10,...]
    :return; slopes over given time
    ''' 
    results = holder()

    results = slopecolumns(prices,periods,['high','low','close'])
    return results

def slopes_old(prices, periods):
    '''
    :param prices; dataframe of OHLC currency data
    :param periods; list of periods for which to compute the function [3,5,10,...]
    :return; slopes over given time
    ''' 
    results = holder()
    
    plot = False
    
    dict = {}
    for i in range(0,len(periods)):
        resdf = pd.DataFrame(columns = ['date','slope_high','slope_low','slope_close'])
        for j in range(periods[i]-1,len(prices)):
            x = np.arange(0,periods[i])
            
            #iterate through columns
            yh = prices.high.iloc[j-periods[i]+1:j+1].values
            yl = prices.low.iloc[j-periods[i]+1:j+1].values
            yc = prices.close.iloc[j-periods[i]+1:j+1].values
            resh = stats.linregress(x,yh)
            resl = stats.linregress(x,yl)
            resc = stats.linregress(x,yc)
            slopeh = resh.slope
            slopel = resl.slope
            slopec = resc.slope

            if (plot == True):
                print(periods[i])
                print(x)
                print(yh)
                print(resh)
                print(prices.index[j])
                
                plt.plot(x,yh,'o',label='original data')
                plt.plot(x,resh.intercept+x*slopeh,'r',label='fitted line')
                plt.show()
            
            resdf = resdf.append({'date':prices.index[j],'slope_high':slopeh,'slope_low':slopel,'slope_close':slopec},ignore_index=True)
            
            
        resdf = resdf.set_index(resdf.date)
        resdf = resdf[['slope_high','slope_low','slope_close']]
        dict[periods[i]] = resdf
    results.df = dict
    return results


def slopecolumns(xprices, periods, scolumns):
    '''
    :param prices; dataframe of OHLC currency data
    :param periods; list of periods for which to compute the function [3,5,10,...]
    :param columns
    :return; slopes over given time
    ''' 
    results = holder()
    
    plot = False
    dict = {}
    for i in range(0,len(periods)):
        resdf = pd.DataFrame(index=xprices.index)
        for k in range(0,len(scolumns)):
            prices = xprices.drop(xprices[xprices[scolumns[k]].isna()].index,axis=0)
            resser = pd.Series(index=prices.index)
            plotcount = 5
            for j in range(periods[i]-1,len(prices)):
                x = np.arange(0,periods[i])
                yh = prices[scolumns[k]].iloc[j-periods[i]+1:j+1].values
                resh = stats.linregress(x,yh)
                slopeh = resh.slope

                if (plot == True and plotcount>0):
                    plotcount -= 1
                    print(periods[i])
                    print(x)
                    print(yh)
                    print(resh)
                    print(prices.index[j])
                
                    plt.plot(x,yh,'o',label='original data')
                    plt.plot(x,resh.intercept+x*slopeh,'r',label='fitted line')
                    plt.show()
            
                
                resser[j] = slopeh
                
            colID = scolumns[k]+'_slope'+str(periods[i])
            resdf[colID] = resser
            

        dict[periods[i]] = resdf
    results.df = dict
    return results


#Detrender
def detrend(prices,method = 'difference',col = 'close'):
    '''
    :param prices; dataframe of OHLC currency data
    :param method; method by which to detrend 'Linear' or 'difference
    :return; the detrended price series
    '''
    if method == 'difference':
        detrended = prices[col].diff()
        # detrended = prices.close[1:]-prices.close[0:-1].values #.values - to get rid of index
    elif method == 'linear':
        x = np.arange(0,len(prices))
        x = x.reshape(-1,1)
        y = prices.close.values
        
        model = LinearRegression()
        model.fit(x,y)
        print("Training set score: {:.2f}".format(model.score(x, y)))
        trend = model.predict(x)
        trend = trend.reshape(len(prices),)
        detrended = prices.close - trend
      
    else:
        print('Wrong method: difference/linear')
    
    return detrended
    
# Fourier Series Expansion Fitting Function
# F = a0+a1*cos(w*x)+b1*sin(w*x)
def fseries(x,a0,a1,b1,w):
    '''
    :param x: the hours (independent variable)
    :param a0: first fourier series coefficient
    :param a1: second fourier series coefficient
    :param b1: third fourier serires coefficient
    :param w: fourier series frequency
    :return: the value of the fourier function
    '''
    
    f = a0+a1*np.cos(w*x)+b1*np.sin(w*x)
    
    return f

# Sine Series Expansion Fitting Function
# F = a0+b1*sin(w*x)
def sseries(x,a0,b1,w):
    '''
    :param x: the hours (independent variable)
    :param a0: first sine series coefficient
    :param b1: third sine serires coefficient
    :param w: sine series frequency
    :return: the value of the sine function
    '''
    
    f = a0+b1*np.sin(w*x)
    
    return f

#Fourier Series Coefficient Calculation Function

def fourier(prices, periods):
    '''
    :param prices; dataframe of OHLC currency data
    :param periods; list of periods for which to compute coefficients [3,5,10,...]
    :param method; method by which to detrend 'Linear' or 'difference
    :return; dict of dateframes containing coefficients for said periods
    '''
    
    results = holder()
    dict = {}
    
    # Option to plot the expansion fit for each iteration
    plot = False
    maxplotcount = 15
    
#     detrended = detrend(prices, method)  
    
    for i in range(0,len(periods)):
        resdf = pd.DataFrame(columns = ['date','a0','a1','b1','w'])
        plotcount = maxplotcount
        for j in range(periods[i] - 1,len(prices)): 
            x = np.arange(0,periods[i])
#             y = detrended.iloc[j - periods[i] + 1:j+1] 
            y = 100 * (np.log(prices.close.iloc[j - periods[i] + 1:j+1])-np.log(prices.close.iloc[j]))
            #TODO: może różnice powinny być podane w procentach?a może nie. a może rożnica logarytmów(done)
            #TOFO: nie wiem w sumie czy to powinno się nazywać detrend. Chodzi o to, aby mieć lokalne wahania, aby na tym obliczyć fouriera.
            # done TODO: 1. zamiast odejmować close[j]-close[j-1] można zrobić tak, że od wszyskich close z okna odejmujemy close z początku okna
            #TODO: 2. inny sposób na detrend: rolling linregression(na danym zakresie=period): = close - prosta linreg - można użyć tego co jest w funkcji slopes lub model.LinearRegression
            
            
            with warnings.catch_warnings():
                warnings.simplefilter('error', OptimizeWarning)
                
                try:
                    res,abc = scipy.optimize.curve_fit(fseries,x,y)

                #except (RuntimeError,OptimizeWarning):
                except:
                    res = np.array([np.NaN,np.NaN,np.NaN,np.NaN])
            if (plot == True and plotcount > 0):
                plotcount -= 1
                print(periods[i])
                print(x)
                print(y)
                print(res)
                print(y.index[periods[i]-1])
                
                xt=np.linspace(0,periods[i]-1,100)
                yt=fseries(xt, res[0],res[1],res[2],res[3])
                
                plt.plot(x,y)
                plt.plot(xt,yt,'r')
                plt.show()
            
            resdf = resdf.append({'date':prices.index[j],'a0':res[0],'a1':res[1],'b1':res[2],'w':res[3]},ignore_index=True)
        warnings.filterwarnings('ignore',category=np.VisibleDeprecationWarning)
        resdf = resdf.set_index(resdf.date)
        resdf = resdf[['a0','a1','b1','w']]
        #resdf = resdf.fillna(0) #df = df.fillna(method='bfill')
        dict[periods[i]]=resdf
    results.df = dict
    return results

#Sine Series Coefficient Calculation Function

def sine(prices, periods, method = 'difference'):
    '''
    :param prices; dataframe of OHLC currency data
    :param periods; list of periods for which to compute coefficients [3,5,10,...]
    :param method; method by which to detrend 'Linear' or 'difference
    :return; dict of dateframes containing coefficients for said periods
    '''
    
    results = holder()
    dict = {}
    
    # Option to plot the expansion fit for each iteration
    plot = False
    maxplotcount = 15
    
#     detrended = detrend(prices, method)
    
    for i in range(0,len(periods)):
        resdf = pd.DataFrame(columns = ['date','a0','b1','w'])
        plotcount = maxplotcount
        for j in range(periods[i] - 1,len(prices)): 
            x = np.arange(0,periods[i])
#             y = detrended.iloc[j - periods[i] + 1:j+1] 
            y = 100 * (np.log(prices.close.iloc[j - periods[i] + 1:j+1])-np.log(prices.close.iloc[j]))
            
            with warnings.catch_warnings():
                warnings.simplefilter('error', OptimizeWarning)
                
                try:
                    res,abc = scipy.optimize.curve_fit(sseries,x,y)

                #except (RuntimeError,OptimizeWarning):
                except:
                    res = np.array([np.NaN,np.NaN,np.NaN])
            if (plot == True and plotcount > 0):
                plotcount -= 1
                print(periods[i])
                print(x)
                print(y)
                print(res)
                
                xt=np.linspace(0,periods[i]-1,100)
                yt=sseries(xt, res[0],res[1],res[2])
                
                plt.plot(x,y)
                plt.plot(xt,yt,'r')
                plt.show()
            
            resdf = resdf.append({'date':prices.index[j],'a0':res[0],'b1':res[1],'w':res[2]},ignore_index=True)
        warnings.filterwarnings('ignore',category=np.VisibleDeprecationWarning)
        resdf = resdf.set_index(resdf.date)
        resdf = resdf[['a0','b1','w']]
        #resdf = resdf.fillna(0) #df = df.fillna(method='bfill')
        dict[periods[i]]=resdf
    results.df = dict
    return results


def atr(prices, periods):
    results = holder()
    dict = {}
    for i in range(0, len(periods)):
        resdf = pd.DataFrame(index=prices.index)
        resdf0 = pd.DataFrame(index=prices.index)
        resdf0['tr1'] = prices['high'] - prices['low']
        resdf0['tr2'] = abs (prices['high'] - prices['close'].shift())
        resdf0['tr3'] = abs (prices['low'] - prices['close'].shift())
        resdf['tr'] = resdf0.max(axis=1)
        resdf['atr'] = resdf.tr.rolling(periods[i]).mean()
        ## slopes
        slopecolumnlist = ['tr','atr']
        sl = slopecolumns(resdf,slopeperiods,slopecolumnlist)
        for kk in range(0,len(slopecolumnlist)):
            for ss in range(0,len(slopeperiods)):
                sscolID = slopecolumnlist[kk]+'_slope' + str(slopeperiods[ss])
                resdf[sscolID]=sl.df[slopeperiods[ss]][sscolID];
        
        dict[periods[i]] = resdf
    
    results.df = dict
    return results


#GARCH
def TSA(prices,periods,order = [0,1,0],plot = False):
    results = holder()
    dict = {}
    
    #res_tup = _get_best_model(resdf.returns.values)
    #order = res_tup[1]
    #order = [1,0,1]

    if order[0] + order[1] <= 0:
        return results
    
    for i in range(0,len(periods)):
        resdf = pd.DataFrame(index=prices.index)
        resdf['log_close'] = np.log(prices.close)
        resdf['returns'] = (resdf.log_close - resdf.log_close.shift())
        resdf.dropna(inplace=True)

        resdf['arima_h1'] = 0
        resdf['garch_h1'] = 0
        resdf['garch_mean'] = 0
        X = resdf.returns.values * 100
        
        for j in range(periods[i] - 1 ,len(resdf)): 
            first_obs = j - periods[i] + 1
            last_obs = j
            TS = X[first_obs:last_obs+1]
            arimah1 = predictARIMAOnSlice(TS,order)
            resdf.iloc[j, resdf.columns.get_loc('arima_h1')]=arimah1/100
            garchh1,garchmean = predictGARCHOnSlice(TS,order)
            resdf.iloc[j, resdf.columns.get_loc('garch_h1')]=garchh1/100
            resdf.iloc[j, resdf.columns.get_loc('garch_mean')]=garchmean

        arima_h1cumsum = resdf.arima_h1.cumsum()
        predictions_log_base = resdf.log_close.iloc[periods[i] - 1]
        predictions_log = arima_h1cumsum.add(predictions_log_base,fill_value=0)
        predictions = np.exp(predictions_log)
        predictions_shift = predictions.shift(1) # predykcja dotyczy następnej linijki
        h1_shift = resdf.garch_h1.shift(1) # predykcja dotyczy następnej linijki
        resdf['arima_prediction'] = predictions
        
        ## slopes
        slopecolumnlist = ['arima_prediction']
        sl = slopecolumns(resdf,slopeperiods,slopecolumnlist)#tu zmienić
        for kk in range(0,len(slopecolumnlist)):
            for ss in range(0,len(slopeperiods)):
                sscolID = slopecolumnlist[kk]+'_slope' + str(slopeperiods[ss])
                resdf[sscolID]=sl.df[slopeperiods[ss]][sscolID];#tu zmienić
        
        dict[periods[i]] = resdf
    
        if plot == True:
#             plt.figure(1)
#             plt.figure(figsize=(12,16))
#             plt.subplot(211)
#             plt.plot(resdf.returns[1+periods[i]:].abs())
#             plt.plot(h1_shift[1+periods[i]:])

#             plt.subplot(212)
#             plt.plot(prices.close[2+periods[i]:],marker='x')
#             plt.plot(predictions_shift[1+periods[i]:],marker='x')
#             plt.show    

            g_mse = mean_squared_error(resdf.returns.iloc[1+periods[i]:].abs(),h1_shift[1+periods[i]:])
            a_mse = mean_squared_error(prices.close[2+periods[i]:],predictions_shift[1+periods[i]:])
            print(order,g_mse,a_mse)
    
    
    results.df = dict
    return results
    
def _get_best_model(TS):

    warnings.filterwarnings('ignore',category=Warning)
    
    best_aic = np.inf 
    best_order = (0,0,0)
    best_mdl = None

    pq_rng =  [1,2,3,4]
    d_rng =  [0,1]
    #pq_rng = [2] 
    #d_rng = [0]
    with warnings.catch_warnings():
        warnings.simplefilter('error', OptimizeWarning)
        for i in pq_rng:
            for d in d_rng:
                for j in pq_rng:
                    try:
                        tmp_mdl = ARIMA(TS, order=(i,d,j)).fit(disp=0)  #method='mle', trend='nc'
                        tmp_aic = tmp_mdl.aic
                        if tmp_aic < best_aic:
                            best_aic = tmp_aic
                            best_order = (i, d, j)
                            best_mdl = tmp_mdl

                    except: 
                        continue
    print('aic: {:6.5f} | order: {}'.format(best_aic, best_order))                    
    return best_aic, best_order, best_mdl    

def predictARIMAOnSlice(TS,order):
    p_ = order[0]
    o_ = order[1]
    q_ = order[2]
    warnings.filterwarnings('ignore',category=Warning)
    with warnings.catch_warnings():
        warnings.simplefilter('error', OptimizeWarning)

        try:
            am = ARIMA(TS, order=(p_, o_, q_)) #method='mle', trend='nc'
            res = am.fit(disp=0)
            forecasts = res.forecast()
            h1 = forecasts[0][0]
        except:
            h1 = np.NaN
    
    if math.isnan(h1):
        h1 = 0
    
    return h1

def predictGARCHOnSlice(TS,order):
    
    p_ = order[0]
    o_ = order[1]
    q_ = order[2]
    warnings.filterwarnings('ignore',category=Warning)
    
    with warnings.catch_warnings():
        warnings.simplefilter('error', OptimizeWarning)

        try:
            am = arch_model(TS, vol='Garch', p=p_,o=o_,q=q_, dist='Normal') # tu może być jeszcze dist='StudentsT'
            res = am.fit(disp='off')
            forecasts = res.forecast()
            h1   = forecasts.variance['h.1']
            h1   = h1[len(TS)-1]
            mean = forecasts.mean['h.1']
            mean = mean[len(TS)-1]
        except:
            h1 = np.NaN
    
    if math.isnan(h1):
        h1 = 0
        mean = 0

    return h1,mean


