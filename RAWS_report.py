import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from requests_html import HTMLSession
from urllib.parse import urljoin
import os

class Report():
    '''
       This class will be used to imports data from RAWS stations. This data is obtained from wrcc.dri.edu.
       BeautifulSoup is used to complete the HTML form to generate the report.
    '''
    def __init__(self,SiteIDs=None):
        self.URL_rep = 'https://raws.dri.edu/cgi-bin/wea_mnsimts.pl?orOBUS' #URL needed to access monthly
            #time series form. This URL is specific to the ORBUS station, but we can actually change 
            #the station hidden variable to any other station
        self.action = '../cgi-bin/wea_mnsimts2.pl' #this is the name of the CGI script that is run 
            #to get the report
            
        self.SiteIDs = SiteIDs
        
        #Year Quarter definition for quarterly slicing
        Q1 = (1,3)
        Q2 = (4,6)
        Q3 = (7,9)
        Q4 = (10,12)

        self.Qdef = [Q1,Q2,Q3,Q4]
        
    def GetIDs(self):
        '''
            Right now this is set up to get all the CA IDs.
        '''
        URL_base = 'https://raws.dri.edu'
        URL_CAs = ['/ncalst.html','/ccalst.html','/scalst.html'] #need to add to this if you want more than
            #CA IDs
        Nlst = len(URL_CAs)
        URL_meta =  'https://wrcc.dri.edu/Monitoring/Stations/station_inventory_show.php?snet=raws&sstate=CA'
            #also need to add in metadata URLs if you want more than CA info
        URL_dates = 'https://wrcc.dri.edu/cgi-bin/rawNAME.pl?ca'
        
        df_syms = [None]*Nlst
        for nn in range(Nlst):
            URL = URL_base + URL_CAs[nn]
            
            session = HTMLSession()
            res = session.get(URL)

            soup = BeautifulSoup(res.html.html, "html.parser")
            
            Body = soup.find('body')
            As = Body.find_all('a')
            Nstn = len(As)-1 #the last one is an email link
            
            StnName = [None]*Nstn
            StnSym = [None]*Nstn
            for ii in range(Nstn):
                StnString = As[ii].text.split()
                StnName[ii] = ''
                NL = 0
                for strings in StnString:
                    if NL == len(StnString)-1:
                        StnName[ii]+=strings
                    else:
                        StnName[ii]+=strings+' '
                    NL+=1
                href = As[ii]['href']
                symstr = href.split('?')[1]
                symstr = list(symstr)
                StnSym[ii] = ''
                for lett in symstr[2:]:
                    StnSym[ii]+=lett
                    
            df_SiteIDs = pd.DataFrame({'Name':StnName,'Symbol':StnSym})
            
            #Make boolean array showing whether or not the site is in CA
            I_CA = np.ones(Nstn).astype(int) == 0 #initialize all as false 

            for ii in range(Nstn):
                String = df_SiteIDs['Name'].iloc[ii].split()
                for words in String:
                    CAbool = words == 'California'
                    if CAbool:
                        I_CA[ii] = True
        
            I_notCA = I_CA == False 
    
            df_SiteIDsCA = df_SiteIDs.drop(df_SiteIDs[I_notCA].index)
            N = len(df_SiteIDsCA)
            NameNoState = [None]*N

            for ii in range(N):
                String = df_SiteIDsCA['Name'].iloc[ii].split()
                Nstr = len(String)
                NameNoState[ii] = ''
                for jj in range(Nstr-1):
                    if jj == (Nstr-2):
                        NameNoState[ii]+=String[jj]
                    else:
                        NameNoState[ii] += String[jj] + ' '
    
            df_SiteIDsCA['Name'] = NameNoState
            
            df_syms[nn] = df_SiteIDsCA
            
        df_SiteIDs = pd.concat(df_syms,ignore_index=True).copy()
        
        #Now get the availability in years of each station
        session = HTMLSession()
        Nstn = len(df_SiteIDs)
        Y_start = [None]*Nstn
        Y_end = [None]*Nstn
        for ii in range(Nstn):
            stn = df_SiteIDs['Symbol'].iloc[ii]
            res = session.get(URL_dates + stn)

            soup = BeautifulSoup(res.html.html, "html.parser")
            Body = soup.find('body')
            Table = Body.find('table')
            Rows = Table.find_all('tr')
            
            if len(Rows) != 0:
                Y_start[ii] = int(Rows[-1].find('td').text)
                Y_end[ii] = int(Rows[0].find('td').text)
            else:
                Y_start[ii] = -9999
                Y_end[ii] = -9999
            
        df_SiteIDs['Y_start'] = Y_start
        df_SiteIDs['Y_end'] = Y_end
        
        #Now add the metadata to the df
        session = HTMLSession()
        res = session.get(URL_meta)

        soup = BeautifulSoup(res.html.html,'html.parser')
        
        datatext = soup.find('pre')
        filename = 'Metadata.txt'
        open(filename,'w').write(datatext.text) #we'll delete this later
        
        #StationMetadata = pd.read_csv(filename,skiprows=7,delimiter='\n',header=None)
        file_meta = open(filename)
        StationMetadata = file_meta.readlines()[7:]
        for ii,entry in enumerate(StationMetadata):
            S = entry.split('\n')
            StationMetadata[ii] = S[0]
        file_meta.close()
        
        os.remove(filename)
        
        Nmeta = len(StationMetadata)
        Name = [None]*Nmeta
        Elev = np.zeros(Nmeta).astype(int)
        Lon = np.zeros(Nmeta).astype(float)
        Lat = np.zeros(Nmeta).astype(float)

        for ii in range(Nmeta):
            String = StationMetadata[ii].split()
    
            if len(String) != 0:
                NL = 0
                for x in String:
                    if x == 'California':
                        ind_CA = NL
                    NL+=1
                if ind_CA == 1:
                    Name[ii] = String[0]
                else:
                    Name[ii] = ''
                    NL = 0
                    for x in String[0:ind_CA]:
                        if NL == (ind_CA-1):
                            Name[ii]+= x
                        else:
                            Name[ii]+= x + ' '
                        NL+=1
                A = String[ind_CA+1]
                A = A.split(',') #there is at least one row with comma in number value, seems to be an error
                Na = len(A)
                if Na == 1:
                    Elev[ii] = int(String[ind_CA+1])
                else:
                    Elev[ii] = -9999 #this is missing data tag for RAWS
                A = String[ind_CA+2]
                A = A.split(',') #there is at least one row with comma in number value, seems to be an error
                Na = len(A)
                if Na == 1:
                    StrL = list(String[ind_CA+2])
                    LatDD = StrL[0:2]
                    LatD =''
                    for x in LatDD:
                        LatD+=x
                    LatMM = StrL[2:4]
                    LatM = ''
                    for x in LatMM:
                        LatM+=x
                    LatSS = StrL[4:6]
                    LatS = ''
                    for x in LatSS:
                        LatS+=x
                    Lat[ii] = float(LatD) + float(LatM)/60 + float(LatS)/3600
                else:
                    Lat[ii] = -9999
                A = String[ind_CA+3]
                A = A.split(',') #there is at least one row with comma in number value, seems to be an error
                Na = len(A)
                if Na == 1:
                    StrL = list(String[ind_CA+3])
                    LonDDD = StrL[0:3]
                    LonD = ''
                    for x in LonDDD:
                        LonD+=x
                    LonMM = StrL[3:5]
                    LonM = ''
                    for x in LonMM:
                        LonM+=x
                    LonSS = StrL[5:7]
                    LonS = ''
                    for x in LonSS:
                        LonS+=x
                    Lon[ii] = -(float(LonD) + float(LonM)/60 + float(LonS)/3600)
                else:
                    Lon[ii] = -9999
        
        Name = np.array(Name)
        Idrop = Name == None
        Name = Name[~Idrop]
        Elev = Elev[~Idrop]
        Lat = Lat[~Idrop]
        Lon = Lon[~Idrop]
        
        #Initialize new df columns as -9999 (missing data value) in case they aren't found
        N_dfCA = len(df_SiteIDs)
        N_meta = len(StationMetadata)

        df_SiteIDs['Elev'] = np.ones(N_dfCA).astype(int)*-9999
        df_SiteIDs['Lat'] = np.ones(N_dfCA).astype(float)*-9999
        df_SiteIDs['Lon'] = np.ones(N_dfCA).astype(float)*-9999

        for ii in range(N_dfCA):
            I = Name == df_SiteIDs.loc[ii,'Name']
            if I.sum() == 1:
                df_SiteIDs.loc[ii,'Elev'] = Elev[I].copy()
                df_SiteIDs.loc[ii,'Lat'] = Lat[I].copy()
                df_SiteIDs.loc[ii,'Lon'] = Lon[I].copy()
                
        self.SiteIDs = df_SiteIDs.copy()
            
        
    def GenHTML(self, save_dir = None,write_file=True,stn='OBUS',smon='10',syea='10',emon='10',eyea='11'):
        '''
            This method generates the HTML page for the station report. Page is saved somewhere locally 
            if write_file=True, and then LoadHTMLReport can later read it in as a dataframe.
        
            If you don't want to locally save the HTML, specify write_file=False and then the HTML
            will be available in self.soup for later use (such as converting to dataframe with LoadHTMLReport)
        
            stn = (str) shorthand code for station
            smon = (str) starting month. '01' = Jan, '12' = Dec.
            syea = (str) starting year. '96' = 1996, '04' = 2004
            emon = (str) ending month
            eyea = (str) ending year
        
            save_dir = path to save directory for html
        
            html file is saved in specified dir, if dir=None save in cd. Naming convention is:
                filename = stn_smon_syea_emon_eyea.html
        '''
        session = HTMLSession()
        
        URL = urljoin(self.URL_rep,self.action)
        
        filename = stn+'_'+smon+'_'+syea+'_'+emon+'_'+eyea+'.html'
        if save_dir != None:
            filename = os.path.join(save_dir,filename)
            
        self.filename = filename
        
        data={}
        data['stn'] = stn
        data['.cgifields'] = 'qST'
        data['qBasic'] = 'ON'
        data['unit'] = 'E'
        data['Ofor'] = 'H'
        data['qc'] = 'Y'
        data['obs'] = 'N'
        data['miss'] = '08'
        data['smon'] = smon
        data['syea'] = syea
        data['emon'] = emon
        data['eyea'] = eyea
        
        res = session.post(URL, data=data)
        
        soup = BeautifulSoup(res.content, "html.parser")
        for a in soup.find_all("a"):
            try:
                a.attrs["href"] = urljoin(url, a.attrs["href"])
            except:
                pass
        
        if write_file:        
            open(filename, "w").write(str(soup))
            
        self.soup = soup
        self.stn = stn
        
    def LoadHTMLReport(self,open_file=True,filename=None):
        '''
            If you are loading from a save HTML page (obatined via GetHTML or other source) this turns it into
            a dataframe. Filename input is needed.
        
            If using HTML from self.soup, specify open_file = False
        '''
        if open_file:
            file = open(filename,'r')
            soup = BeautifulSoup(file,'html.parser')
            stn = filename.split('_')[0]
        else:
            soup = self.soup
            stn = self.stn
            
        if self.SiteIDs is None:
            self.GetIDs() 
        
        I_stn = self.SiteIDs['Symbol'] == stn
        Elev = self.SiteIDs['Elev'].loc[I_stn].values[0]
        Lat = self.SiteIDs['Lat'].loc[I_stn].values[0]
        Lon = self.SiteIDs['Lon'].loc[I_stn].values[0]
        
        Table = soup.find('table')
        Rows = Table.find_all('tr')
        Nrows = len(Rows)
        
        #First row has data type names (first column is a blank, that is the date column)
        Cols = Rows[0].find_all('td')
        Ncols = len(Cols)

        DataNames = [None]*(Ncols-1)
        DataColSpan = np.zeros(Ncols-1).astype(int)
        NL = 0
        for ii in range(1,Ncols):
            A = Cols[ii].text
            A = A.split()
            String=''
            NLsub=0
            for x in A:
                if NLsub == 0:
                    String += x
                else:
                    String += ' ' + x
                NLsub+=1
            DataNames[NL] = String
            DataColSpan[NL] = Cols[ii]['colspan']
            NL+=1
            
        #Second row has units of the datatypes
        Cols = Rows[1].find_all('td')
        DataUnits = [None]*(Ncols-1)
        NL=0
        for ii in range(1,Ncols):
            A = Cols[ii].text
            A = A.split()
            String = ''
            for x in A: #this gets rid of the newline characters
                String+=x
            DataUnits[NL] = String
            NL+=1
            
        #The third row is the datatype subtype (like if it's an average or maximum observation)
        Cols = Rows[2].find_all('td')
        Ncols = len(Cols)
        DataSubType = [None]*(Ncols-1)

        NL=0
        for ii in range(1,Ncols):
            A = Cols[ii].text
            A = A.split()
            String = ''
            NLsub=0
            for x in A:
                if NLsub == 0:
                    String+=x
                else:
                    String+=' '+x
                NLsub+=1
            DataSubType[NL] = String
            NL+=1
            
        #The remainder of rows are the data, each row for a new month-year
        Ndat = Nrows-3
        Ncol = DataColSpan.sum()+1

        Month = np.zeros(Ndat).astype(int)
        Year = np.zeros(Ndat).astype(int)
        Data = np.zeros((Ndat,Ncol-1))

        NL = 0
        for ii in range(3,Nrows):
            Cols = Rows[ii].find_all('td')
            for jj in range(Ncol):
                if jj == 0:
                    A = Cols[jj].text
                    A = A.split('/')
                    Month[NL] = A[0]
                    Year[NL] = A[1]
                else:
                    Data[NL,jj-1] = float(Cols[jj].text)
            NL+=1
        
        df = pd.DataFrame()
        df['Station'] = [stn]*Ndat
        df['Elev'] = [Elev]*Ndat
        df['Lat'] = [Lat]*Ndat
        df['Lon'] = [Lon]*Ndat
        df['Month'] = Month
        df['Year'] = Year

        Nmain = len(DataNames)
        Name = [None]*(Ncol-1)
        NL = 0
        for ii in range(Nmain):
            MainName = DataNames[ii]+', '+DataUnits[ii]
            for jj in range(DataColSpan[ii]):
                Name[NL] = MainName+', '+DataSubType[NL]
                NL+=1
                
        for ii in range(Ncol-1):
            df[Name[ii]] = Data[:,ii]
            
        self.df_rep = df
        
    def GetReport(self,stn='OBUS',TimeInput={'smon':'01','syea':'15','emon':'12','eyea':'19'}):
        '''
            This gets the HTML page from the CGI script webform and converts it to a dataframe.
            Dictionary input for time values as laid out in default values.
        '''
        
        self.GenHTML(write_file=False,stn=stn,\
            smon=TimeInput['smon'],syea=TimeInput['syea'],emon=TimeInput['emon'],eyea=TimeInput['eyea'])
        
        self.LoadHTMLReport(open_file=False)
        
        #now the dataframe is contained in self.df_rep
        
    def QuarterlyReslice(self,df=None,YearRange=(1990,2019),inplace=False):
        '''
            Takes monthly timeseries and reslices it into quarters as opposed to months.
            The months in each quarter are averaged over.
        ''' 
        if df is None:
            df = self.df_rep
            
        Nq = len(self.Qdef)
        
        Years = np.arange(YearRange[0],YearRange[1]+1,1).astype(int)
        Nyear = len(Years)
        
        ColNames = df.columns
        
        IndAvStart = 6 #average over columns from this index onward in ColNames
        ColsAv = ColNames[IndAvStart:]
        Nav = len(ColsAv)

        ColsInit = ColNames[:4]
        Ninit = len(ColsInit)
        
        DictDFInit = {}
        for ii in range(Ninit):
            DictDFInit[ColsInit[ii]] = [df[ColsInit[ii]].iloc[0]]
            
        for ii in range(Nyear):
            I_year = df['Year'] == Years[ii]
            for jj in range(Nq):
                I_q =  (df['Month'] >= self.Qdef[jj][0]) & (df['Month'] <= self.Qdef[jj][1])
                I_slice = I_q & I_year
                df_sub = df.loc[I_slice]
                ColDataAv = np.zeros(Nav)
                DictDF = DictDFInit
                DictDF['Q'] = jj+1
                DictDF['Year'] = Years[ii]
                for kk in range(Nav):
                    #need to deal with both empty entries and empty values that are set to -9999
                    datavals = df_sub[ColsAv[kk]].values
                    IndKeep = []
                    NL=0
                    for vals in datavals:
                        if vals != -9999:
                            IndKeep.append(NL)
                        NL+=1
                    datavals = datavals[IndKeep]
            
                    if len(datavals) == 0:
                        ColDataAv[kk] = -9999
                    else:
                        ColDataAv[kk] = np.mean(datavals)
                    DictDF[ColsAv[kk]] = [ColDataAv[kk]]
                if (ii==0) & (jj==0):
                    df_quarter = pd.DataFrame(DictDF)
                else: 
                    df_quarter = pd.concat([df_quarter,pd.DataFrame(DictDF)])
                    
        df_quarter.set_index(np.arange(df_quarter.shape[0]),inplace=True)

        if inplace:
            self.df_rep = df_quarter
        else:
            return df_quarter
            
    def HistavNorm(self,df=None,inplace=False):
        '''
            This takes QuarterlyResliced dataframes and normalizes them to an historical
            average for each quarter. Units of data are then just fractional change from
            the average of the whole dataset (per quarter over all years)
        '''
        if df is None:
            df = self.df_rep

        Nq = len(self.Qdef)
    
        #Years = np.arange(YearRange[0],YearRange[1]+1,1).astype(int)
        Years = np.unique(df['Year'].values)
        Nyear = len(Years)
    
        ColNames = df.columns
    
        IndAvStart = 6 #average over columns from this index onward in ColNames
        ColsAv = ColNames[IndAvStart:]
        Nav = len(ColsAv)

        ColsInit = ColNames[:4]
        Ninit = len(ColsInit)
        
        DictDFInit = {}
        for ii in range(Ninit):
            DictDFInit[ColsInit[ii]] = [df[ColsInit[ii]].iloc[0]]
            
        for ii in range(Nq):
            I_q = df['Q'] == ii+1
            ColAvData = np.zeros(Nav)
            DictDF = DictDFInit
            DictDF['Q'] = [ii+1]
            DictDF['Y_start'] = [Years[0]]
            DictDF['Y_end'] = [Years[-1]]
            for jj in range(Nav):
                #again, need to treat -9999 values (no measurement), should be no empty entries now
                datavals = df[ColsAv[jj]].loc[I_q].values
                IndKeep = []
                NL=0
                for val in datavals:
                    if val != -9999:
                        IndKeep.append(NL)
                    NL+=1
                datavals = datavals[IndKeep]
                AvVal = np.mean(datavals)
                DictDF[ColsAv[jj]] = [AvVal]
            if (ii==0):
                df_histAv = pd.DataFrame(DictDF)
            else:
                df_histAv = pd.concat([df_histAv,pd.DataFrame(DictDF)])
                
        df_norm = df.copy()
        for ii in range(Nq):
            I_q = df_norm['Q'] == ii+1
            for jj in range(Nyear):
                I_year = df_norm['Year'] == Years[jj]
                I = I_q & I_year
                for kk in range(Nav):
                    HistVal = df_histAv[ColsAv[kk]].iloc[ii]
                    Val = df_norm.loc[I,ColsAv[kk]].values
                    if Val != -9999:
                        df_norm.loc[I,ColsAv[kk]] = (Val - HistVal)/HistVal
            
        #now just delete the unit portion of the column name since this is now unitless
        for ii in range(Nav):
            StrList = ColsAv[ii].split(',')
            NewName = StrList[0]+','+StrList[2]
            df_norm = df_norm.rename(columns={ColsAv[ii]:NewName}).copy()
   
        RenameDict = {}
        for ii in range(Nav):
            StrList = ColsAv[ii].split(',')
            NewName = StrList[0]+','+StrList[2]
            RenameDict[ColsAv[ii]] = NewName

        df_norm = df_norm.rename(columns=RenameDict)
        
        if inplace:
            self.df_rep = df_norm
        else:
            return df_norm