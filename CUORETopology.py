import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import math
import ast
import seaborn as sns
sns.set_style("whitegrid")
import shapely as shapely
from shapely.geometry import LineString
import scipy.special as sc
from numba_stats import norm, binom
from mpl_toolkits.axes_grid1 import make_axes_locatable
from iminuit import cost, Minuit
import scipy.integrate as integrate

class TopologyAnalysis:
    def __init__(self, config):
        self.dataframe = pd.read_csv(config['DataFile'], sep = ',', header=0)
        self.multipllicity = config['Multiplicity']
        self.energyRange = config['EnergyRange']
        self.invalidChannels = config['InvalidChannels']
        self.energyDist = config['EnergyDist']
        self.bias = config['DatasetBias']
        self.res = config['DatasetRes']
        
    def ChannelRemoval(self, dataframe):
        df = pd.read_csv(self.invalidChannels, sep=',')
        datasetlist = np.arange(3801, 3829)

        # Create a dictionary mapping each dataset to its channel list
        channel_map = df.groupby('Dataset')['Channels'].apply(list).to_dict()

        channel = []

        # Loop through the filtered datasets in dataframe_single
        for i in datasetlist:
            # Get the channel list for the dataset from the pre-built dictionary
            channel_list = channel_map.get(i, [])
            
            # Filter only rows in dataframe_single where the 'Dataset' equals the current dataset
            relevant_rows = dataframe[dataframe['Dataset'] == i]
            
            # Check for channel matches using vectorized apply with any()
            matches = relevant_rows.index[
                relevant_rows.iloc[:, 10].apply(lambda x: any(ch in channel_list for ch in x))
            ].tolist()
            
            # Add matching indices to the channel list
            channel.extend(matches)
            
        dataframe_reduced = dataframe.drop(channel)
        dataframe_reduced=dataframe_reduced.reset_index(drop=True)
            
        return dataframe_reduced
    
    
    def SingleEscapeSort(self):
        dataframe = self.dataframe.drop('Unnamed: 0', axis=1)
        
        col = dataframe.pop("NTD")
        col2 = dataframe.pop('EnergyV')
        col3 = dataframe.pop('ChannelV')
        col4 = dataframe.pop('Neighbors')
        dataframe.insert(9, col2.name, col2)
        dataframe.insert(15, col.name, col)
        dataframe.insert(10, col3.name, col3)
        dataframe.insert(14, col4.name, col4)

        dataframe3 = dataframe[(dataframe['Multiplicity'] == self.multipllicity)]
        dataframe3['EnergyV']=dataframe3['EnergyV'].apply(ast.literal_eval)
        dataframe3['ChannelV']=dataframe3['ChannelV'].apply(ast.literal_eval)
        dataframe3['NTD']=dataframe3['NTD'].apply(ast.literal_eval)
        dataframe3['Neighbors']=dataframe3['Neighbors'].apply(ast.literal_eval)

        rid3 = []
        dataframe3=dataframe3.reset_index(drop=True)
        for i in range(len(dataframe3)):
            if self.multipllicity ==2:
                if (min(self.energyRange)<= dataframe3.iloc[i,9][1]< max(self.energyRange)):
                    rid3.append(i)
            if self.multipllicity==3:
                if (min(self.energyRange)<= dataframe3.iloc[i,9][2]< max(self.energyRange)):
                    rid3.append(i)
                
        dataframe_2100 = dataframe3[dataframe3.index.isin(rid3)]
        dataframe_2100=dataframe_2100.reset_index(drop=True)

        return dataframe_2100
    
    def DoubleEscapeSort(self):
        # dataframe = dataframe.drop('10', axis=1)
        dataframe = self.ChannelRemoval(self.dataframe.drop('Unnamed: 0', axis=1))
        
        col = dataframe.pop("NTD")
        col2 = dataframe.pop('EnergyV')
        col3 = dataframe.pop('ChannelV')
        col4 = dataframe.pop('Neighbors')
        dataframe.insert(9, col2.name, col2)
        dataframe.insert(15, col.name, col)
        dataframe.insert(10, col3.name, col3)
        dataframe.insert(14, col4.name, col4)

        dataframe3 = dataframe[(dataframe['Multiplicity'] == 3)]
        dataframe3['EnergyV']=dataframe3['EnergyV'].apply(ast.literal_eval)
        dataframe3['ChannelV']=dataframe3['ChannelV'].apply(ast.literal_eval)
        dataframe3['NTD']=dataframe3['NTD'].apply(ast.literal_eval)
        dataframe3['Neighbors']=dataframe3['Neighbors'].apply(ast.literal_eval)

        rid3 = []
        dataframe3=dataframe3.reset_index(drop=True)
        for i in range(len(dataframe3)):
            if min(self.energyRange)<= dataframe3.iloc[i,9][1]< max(self.energyRange) and min(self.energyRange)<= dataframe3.iloc[i,9][0]< max(self.energyRange):
                rid3.append(i)

        dataframe3_double = dataframe3[dataframe3.index.isin(rid3)]
        # dataframe3_2100['Neighbors']=dataframe3_2100['Neighbors'].apply(ast.literal_eval)
        # dataframe3_2100['ChannelV']=dataframe3_2100['ChannelV'].apply(ast.literal_eval)
        dataframe3_double=dataframe3_double.reset_index(drop=True)
        return dataframe3_double
    
    def NTD_Sort(self, dataframe_1tonne_single):
        
        NTD_data = []
        no_NTD_data = []

        NTD_data = []

        for i in range(len(dataframe_1tonne_single)):
            if (dataframe_1tonne_single.iloc[i,10][0] in dataframe_1tonne_single.iloc[i,17]) or (dataframe_1tonne_single.iloc[i,10][1] in dataframe_1tonne_single.iloc[i,17]):
                NTD_data.append(i) 
            else:
                no_NTD_data.append(i)

        NTD_data = dataframe_1tonne_single[dataframe_1tonne_single.index.isin(NTD_data)]
        NTD_data=NTD_data.reset_index(drop=True)

        no_NTD_data = dataframe_1tonne_single[dataframe_1tonne_single.index.isin(no_NTD_data)]
        no_NTD_data=no_NTD_data.reset_index(drop=True)
        
        return NTD_data, no_NTD_data
    
    def M2Topology_Sort(self, no_NTD_data):
        F_NTD = []
        E_NTD = []
        N_NTD = []
        C_NTD = []
        O_NTD = []
        for i in range(len(no_NTD_data)):
            if no_NTD_data.iloc[i,14][0][0] == 'F':
                F_NTD.append(i)
            elif no_NTD_data.iloc[i,14][0][0] == 'E':
                E_NTD.append(i)
            elif no_NTD_data.iloc[i,14][0][0] == 'N':
                N_NTD.append(i)
            elif no_NTD_data.iloc[i,14][0][0] == 'C':
                C_NTD.append(i)
            else:
                O_NTD.append(i)
                
        F_NTD = no_NTD_data[no_NTD_data.index.isin(F_NTD)]
        F_NTD=F_NTD.reset_index(drop=True)
        
        E_NTD = no_NTD_data[no_NTD_data.index.isin(E_NTD)]
        E_NTD=E_NTD.reset_index(drop=True)
        
        N_NTD = no_NTD_data[no_NTD_data.index.isin(N_NTD)]
        N_NTD=N_NTD.reset_index(drop=True)
        
        C_NTD = no_NTD_data[no_NTD_data.index.isin(C_NTD)]
        C_NTD=C_NTD.reset_index(drop=True)
        
        return F_NTD, E_NTD, C_NTD, N_NTD
    
    def M3Topology_Sort(self, no_NTD_data):
        FC = []
        CE = []
        NC = []
        FN = []
        NE = []
        FF = []
        CC = []
        EE = []
        NN = []
        FE = []
        O = []

        for i in range(len(no_NTD_data)):
            if no_NTD_data.iloc[i,14][0][0] + no_NTD_data.iloc[i,14][1][0] == 'CF' or no_NTD_data.iloc[i,14][0][0] + no_NTD_data.iloc[i,14][1][0] == 'FC':
                FC.append(i)
            elif no_NTD_data.iloc[i,14][0][0] + no_NTD_data.iloc[i,14][1][0] == 'CE' or no_NTD_data.iloc[i,14][0][0] + no_NTD_data.iloc[i,14][1][0] == 'EC':
                CE.append(i)
            elif no_NTD_data.iloc[i,14][0][0] + no_NTD_data.iloc[i,14][1][0] == 'NC' or no_NTD_data.iloc[i,14][0][0] + no_NTD_data.iloc[i,14][1][0] == 'CN':
                NC.append(i)
            elif no_NTD_data.iloc[i,14][0][0] + no_NTD_data.iloc[i,14][1][0] == 'FN' or no_NTD_data.iloc[i,14][0][0] + no_NTD_data.iloc[i,14][1][0] == 'NF':
                FN.append(i)
            elif no_NTD_data.iloc[i,14][0][0] + no_NTD_data.iloc[i,14][1][0] == 'NE' or no_NTD_data.iloc[i,14][0][0] + no_NTD_data.iloc[i,14][1][0] == 'EN':
                NE.append(i)
            elif no_NTD_data.iloc[i,14][0][0] + no_NTD_data.iloc[i,14][1][0] == 'FF':
                FF.append(i)
            elif no_NTD_data.iloc[i,14][0][0] + no_NTD_data.iloc[i,14][1][0] == 'CC':
                CC.append(i)
            elif no_NTD_data.iloc[i,14][0][0] + no_NTD_data.iloc[i,14][1][0] == 'EE':
                EE.append(i)
            elif no_NTD_data.iloc[i,14][0][0] + no_NTD_data.iloc[i,14][1][0] == 'NN':
                NN.append(i)
            elif no_NTD_data.iloc[i,14][0][0] + no_NTD_data.iloc[i,14][1][0] == 'FE' or no_NTD_data.iloc[i,14][0][0] + no_NTD_data.iloc[i,14][1][0] == 'EF':
                FE.append(i)
            else:
                O.append(i)

        FC = no_NTD_data[no_NTD_data.index.isin(FC)]
        CE = no_NTD_data[no_NTD_data.index.isin(CE)]
        NC = no_NTD_data[no_NTD_data.index.isin(NC)]
        FN = no_NTD_data[no_NTD_data.index.isin(FN)]
        NE = no_NTD_data[no_NTD_data.index.isin(NE)]
        FF = no_NTD_data[no_NTD_data.index.isin(FF)]
        CC = no_NTD_data[no_NTD_data.index.isin(CC)]
        EE = no_NTD_data[no_NTD_data.index.isin(EE)]
        NN = no_NTD_data[no_NTD_data.index.isin(NN)]
        FE = no_NTD_data[no_NTD_data.index.isin(FE)]

        FC =FC.reset_index(drop=True)
        CE =CE.reset_index(drop=True)
        NC =NC.reset_index(drop=True)
        FN =FN.reset_index(drop=True)
        NE =NE.reset_index(drop=True)
        FF =FF.reset_index(drop=True)
        CC =CC.reset_index(drop=True)
        EE =EE.reset_index(drop=True)
        NN =NN.reset_index(drop=True)
        FE =FE.reset_index(drop=True)
        return FC, CE, NC, FN, NE, FF, CC, EE, NN, FE, O
    
    def PrimarySecondarySplitSingleEscape(self,dataframe_1tonne_double):
        total_2100 = []
        total_511 = []

        if self.multipllicity ==2:
            for i in range(len(dataframe_1tonne_double)):
                total_511.append(dataframe_1tonne_double.iloc[i,9][0])
                total_2100.append(dataframe_1tonne_double.iloc[i,9][1])
        elif self.multipllicity ==3:
            for i in range(len(dataframe_1tonne_double)):
                total_511.append(dataframe_1tonne_double.iloc[i,9][0]+dataframe_1tonne_double.iloc[i,9][1])
                total_2100.append(dataframe_1tonne_double.iloc[i,9][2])
            
        return np.asarray(total_2100), np.asarray(total_511)
    
    def PrimarySecondarySplitDoubleEscape(self, dataframe_1tonne_double):
        total_2100 = []
        total_511 = []
        
        for i in range(len(dataframe_1tonne_double)):
            total_511.append(dataframe_1tonne_double.iloc[i,9][0])
            total_511.append(dataframe_1tonne_double.iloc[i,9][1])
            total_2100.append(dataframe_1tonne_double.iloc[i,9][2])
            
        return np.asarray(total_2100), np.asarray(total_511)
        
        
    def Triple_Gaussian(self, x, mu, mu2,mu3, sigma,sigma3, A, c1, c2, c3):
        E0 = min(x)
        E1=max(x)
        Ec = (E0+E1)/2
        p = (1+A*(x-Ec))/(E1-E0)
        return (1-c1-c2-c3)*norm.pdf(x,mu,sigma)+c2*norm.pdf(x,mu2, sigma)+c3*norm.pdf(x,mu3,sigma3)+c1*p
    
    def TripleGaussian(self, x,mu1,mu2,mu3, sigma, c2,c3):
        return (1-c2-c3)*norm.pdf(x,mu1,sigma)+c2*norm.pdf(x,mu2,sigma)+c3*norm.pdf(x,mu3,sigma)
    
    def Double_Gaussian(self, x, mu, mu2, sigma, A, f, d):
        E0 = min(x)
        E1=max(x)
        Ec = (E0+E1)/2
        p = (1+A*(x-Ec))/(E1-E0)
        return (1-f-d)*norm.pdf(x,mu, sigma)+d*norm.pdf(x,mu2, sigma)+f*p
    
    def Chi2(self, data, arr, bin_num):
        hist, bins = np.histogram(data, bin_num)
        bin_width = (max(data) - min(data))/bin_num

        bin_centers = (bins[:-1] + bins[1:]) / 2
        
        non_zero_indices = hist > 0
        filtered_hist = hist[non_zero_indices]
        filtered_bin_centers = bin_centers[non_zero_indices]

        if(self.energyDist == 511):
            fitted_pdf = self.NormGaussianExp(filtered_bin_centers, *arr)*len(data)*bin_width
        elif(self.energyDist == 2100):
            fitted_pdf = self.Double_Gaussian(filtered_bin_centers, *arr)*len(data)*bin_width
        else:
            fitted_pdf = self.NormGaussianExp(filtered_bin_centers, *arr)*len(data)*bin_width

        chi2 = np.sum((filtered_hist-fitted_pdf)**2/fitted_pdf)
        
        return chi2
    
    def PValue(self,dof, chi2):
        return 1-(sc.gammainc(dof/2, chi2/2))
    
    def Residual_Plot(self, data, arr, bin_num):
        hist, bins = np.histogram(data, bin_num)
        bin_width = (max(data) - min(data))/bin_num

        bin_centers = (bins[:-1] + bins[1:]) / 2
        
        non_zero_indices = hist > 0
        filtered_hist = hist[non_zero_indices]
        filtered_bin_centers = bin_centers[non_zero_indices]
        if(self.energyDist == 511):
            fitted_pdf = self.NormGaussianExp(filtered_bin_centers, *arr)*len(data)*bin_width
        elif(self.energyDist == 2100):
            fitted_pdf = self.Double_Gaussian(filtered_bin_centers, *arr)*len(data)*bin_width
        else:
            fitted_pdf = self.NormGaussianExp(filtered_bin_centers, *arr)*len(data)*bin_width
        
        return filtered_bin_centers, (filtered_hist - fitted_pdf)/np.sqrt(filtered_hist), np.sqrt(filtered_hist)
    
    def Triple_GuassianMinimizer(self, data, title, initial_parms, bin_num):
        E1=max(data)
        E0=min(data)
        
        Ec = (E1+E0)/2
        DE = (E1-E0)/2
        bound_A = 1/(Ec-E0)
        
        c = cost.UnbinnedNLL(data, self.Triple_Gaussian)

        m = Minuit(c,*initial_parms)
        m.limits['c1', 'c2', 'c3'] = (0, 1)
        m.limits['mu','sigma', 'mu2', 'mu3', 'sigma3'] = (0, None)
        m.limits["A"] = (-bound_A, bound_A)
        m.migrad()
        m.hesse()

        x_values = np.linspace(min(data), max(data), 1000)
        bin_width = (max(data) - min(data))/bin_num

        fig = plt.figure()
        ax=fig.add_axes((.1,.3,.8,.6))
        
        props = dict(boxstyle='round', facecolor='whitesmoke', alpha=0.5)
        text = '\n'.join(( r'$\Delta\chi^2/ndf= $' + str(np.round(self.Chi2(data, m.values, bin_num), decimals=2)) +'/' +str(bin_num - (self.Triple_Gaussian.__code__.co_argcount -1), ),
                        r'$P(ndf,\chi^2)=%.2f$' % (self.PValue(bin_num - (self.Triple_Gaussian.__code__.co_argcount -1),self.Chi2(data, m.values, bin_num)), ),
            'Entries: ' + (str(len(data))),
            r'$\mu=$ ' + str(np.round(m.values[0], decimals = 2) ) + '+-'+ str(np.round(m.errors[0], decimals = 2) ),
            r'$\mu_2=$ ' + str(np.round(m.values[1], decimals = 2) ) + '+-'+ str(np.round(m.errors[1], decimals = 2) ),
            r'$\mu_3=$ ' + str(np.round(m.values[2], decimals = 2) ) + '+-'+ str(np.round(m.errors[2], decimals = 2) ),
            r'$\sigma=$ ' + str(np.round(m.values[3], decimals = 2) ) + '+-'+ str(np.round(m.errors[3], decimals = 2) ),
            #r'$\sigma_2=$ ' + str(np.round(m.values[4], decimals = 2) ) + '+-'+ str(np.round(m.errors[3], decimals = 2) ),
            r'$\sigma_3=$ ' + str(np.round(m.values[4], decimals = 2) ) + '+-'+ str(np.round(m.errors[4], decimals = 2) ),
            r'$A=$ ' + str(np.format_float_scientific(m.values[5], precision = 2) ) + '+-'+ str(np.format_float_scientific(m.errors[5], precision = 2) ),
            #r'$p_2=$ ' + str(np.format_float_scientific(m.values[6], precision = 2) ) + '+-'+ str(np.format_float_scientific(m.errors[6], precision = 2) ),
            r'$c_1=$ ' + str(np.format_float_scientific(m.values[6], precision = 2) ) + '+-'+ str(np.format_float_scientific(m.errors[6], precision = 2) ),
            r'$c_2=$ ' + str(np.format_float_scientific(m.values[7], precision = 2) ) + '+-'+ str(np.format_float_scientific(m.errors[7], precision = 2) ),
            r'$c_3=$ ' + str(np.format_float_scientific(m.values[8], precision = 2) ) + '+-'+ str(np.format_float_scientific(m.errors[8], precision = 2) )
            ))
        
        mu,mu2, mu3, sigma, sigma3, A, c1,c2, c3 = m.values
        mu_err,mu2_err, mu3_err, sigma_err, sigma3_err, Aerr,c1_err,c2_err, c3_err = m.errors

        ax.hist(data, bins = bin_num , histtype='step');
        ax.plot(x_values, self.Triple_Gaussian(x_values,*m.values)*len(data)*bin_width,label='Fitted PDF', color='#000272')
        ax.plot(x_values, (1-c1-c2-c3)*norm.pdf(x_values,mu, sigma)*len(data)*bin_width, label='Gaussian 1', linestyle='--', color='#A72693')
        ax.plot(x_values, c2*norm.pdf(x_values,mu2, sigma)*len(data)*bin_width, label='Gaussian 2', linestyle='--', color='#02383C')
        ax.plot(x_values, c3*norm.pdf(x_values,mu3,sigma3)*len(data)*bin_width, label='Gaussian 3', linestyle='--', color='#BD512F')
        ax.plot(x_values, c1*((1+A*(x_values-Ec))/(E1-E0))*len(data)*bin_width, label='Linear Bckg', linestyle='--', color='#615DEC')
        # ax.plot(x_values, (sc.erfc((x_values-mu)/(np.sqrt(2)*sigma)))/((mu-E0)*sc.erfc((E0-mu)/(np.sqrt(2)*sigma))+(E1-mu)*sc.erfc((E1-mu)/(np.sqrt(2)*sigma))+np.sqrt(2/np.pi)*sigma*(np.exp(-(mu-E0)**2/(2*sigma**2))-np.exp(-(mu-E1)**2/(2*sigma**2))))*len(data)*bin_width, linestyle='--', label='Step Func')
        ax.legend(loc='upper right')
        ax.text(0.03, 0.95, text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=props)
        ax.set_title(title)

        ax2=fig.add_axes((.1,.08,.8,.2))        
        ax2.errorbar(self.Residual_Plot(data, m.values, bin_num)[0],self.Residual_Plot(data, m.values, bin_num)[1], fmt='o',markersize=4)
        ax2.axhline(0, linestyle='--', color='crimson')
        
        return mu, mu_err, mu2, mu2_err, mu3, mu3_err, sigma, sigma_err, sigma3, sigma3_err
    
    def Triple_GuassianMinimizerTopology(self,data, initial_parms,topologies):
        fitarr = []
        errorarr = []
        PDFs = []
        guass1 = []
        guass2 = []
        guass3 = []
        linear = []
        xvals = []
        bin_num = []
        bin_width = []

        for i in range(len(data)):
            E0 = min(data[i])
            E1 = max(data[i])
            
            Ec = (E0+E1)/2
            bound_A = 1/(Ec-E0)
            
            c = cost.UnbinnedNLL(data[i], self.Triple_Gaussian)

            m = Minuit(c,*initial_parms[i])
            m.limits['c1', 'c2', 'c3'] = (0, 1)
            m.limits['mu','sigma', 'mu2', 'mu3', 'sigma3'] = (0, None)
            m.limits["A"] = (-bound_A, bound_A)
            m.migrad()
            m.hesse()
            
            fitarr.append(m.values)
            errorarr.append(m.errors)
            xvals.append(np.linspace(min(data[i]), max(data[i]), 1000))
            bin_num.append(2*int(E1-E0))
            
            bin_width.append((E1-E0)/(2*int(E1-E0)))

        chisq = []
        pvals = []
        means = []
        means2 = []
        means3= []
        sigmas = []
        # sigmas2 = []
        sigmas3 = []
        A = []
        c1 = []
        c2 = []
        c3 = []

        meanserr = []
        means2err = []
        means3err = []
        sigmaserr = []
        # sigmas2err = []
        sigmas3err = []
        Aerr = []
        c1err = []
        c2err = []
        c3err = []
            
        for i in range(len(fitarr)):
            E0 = min(data[i])
            E1 = max(data[i])
            Ec = (E0+E1)/2
            PDFs.append(self.Triple_Gaussian(xvals[i],*fitarr[i])*len(data[i])*bin_width[i])
            chisq.append(self.Chi2(data[i], fitarr[i], bin_num[i]))
            pvals.append(self.PValue(bin_num[i]-(self.Triple_Gaussian.__code__.co_argcount-1), chisq[i]))
            
            guass1.append((1-fitarr[i][7]-fitarr[i][8]-fitarr[i][6])*norm.pdf(xvals[i],fitarr[i][0], fitarr[i][3])*len(data[i])*bin_width[i])
            guass2.append(fitarr[i][7]*norm.pdf(xvals[i],fitarr[i][1],fitarr[i][3])*len(data[i])*bin_width[i])
            guass3.append(fitarr[i][8]*norm.pdf(xvals[i],fitarr[i][2], fitarr[i][4])*len(data[i])*bin_width[i])
            linear.append(fitarr[i][6]*((1+fitarr[i][5]*(xvals[i]-Ec))/(E1-E0))*len(data[i])*bin_width[i])
            
            means.append(fitarr[i][0])
            means2.append(fitarr[i][1])
            means3.append(fitarr[i][2])
            sigmas.append(fitarr[i][3])
            # sigmas2.append(fitarr[i][4])
            sigmas3.append(fitarr[i][4])
            A.append(fitarr[i][5])
            c1.append(fitarr[i][6])
            c2.append(fitarr[i][7])
            c3.append(fitarr[i][8])
            
            meanserr.append(errorarr[i][0])
            means2err.append(errorarr[i][1])
            means3err.append(errorarr[i][2])
            sigmaserr.append(errorarr[i][3])
            # sigmas2err.append(errorarr[i][4])
            sigmas3err.append(errorarr[i][4])
            Aerr.append(errorarr[i][5])
            c1err.append(errorarr[i][6])
            c2err.append(errorarr[i][7])
            c3err.append(errorarr[i][8])
            
            
            
        data = { 'Topologies': topologies,
            'Bin Num': bin_num,
        'Histogram Data': data,
            'X-axis': xvals,
            'Fit': PDFs,
            'Guass1': guass1,
            'Guass2': guass2,
            'Guass3': guass3,
            'linear': linear,
            'Means': means,
            'Means2': means2,
            'Means3': means3,
            'Sigmas': sigmas,
            # 'Sigmas2': sigmas2,
            'Sigmas3': sigmas3,
            'A':A,
            'c1': c1, 
            'c2': c2,
            'c3': c3,  
            'Chi2': chisq,
            'p-value': pvals, 
            'Meanserr': meanserr,
            'Means2err': means2err,
            'Means3err': means3err,
            'Sigmaserr': sigmaserr,
            # 'Sigmas2err': sigmas2err,
            'Sigmas3err': sigmas3err,
            'Aerr':Aerr,
            'c1err': c1err,
            'c2err': c2err,
            'c3err': c3err,
            }

        df = pd.DataFrame(data)
            
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))
        props = dict(boxstyle='round', facecolor='whitesmoke')
        for i, ax in enumerate(axs.flatten()):
            if i < len(df):
                divider = make_axes_locatable(ax)
                
                text = '\n'.join(( r'$\Delta\chi^2/ndf= $' + str(np.round(df['Chi2'][i], decimals=2)) +'/' +str(df['Bin Num'][i] - (self.Triple_Gaussian.__code__.co_argcount -1), ),
                        r'$P(ndf,\chi^2)=%.2f$' % (df['p-value'][i], ),
            'Entries: ' + (str(len(df['Histogram Data'][i]))),
            r'$\mu=$ ' + str(np.round(df['Means'][i], decimals = 2) ) + '+-'+ str(np.round(df['Meanserr'][i], decimals = 2) ),
            r'$\mu_2=$ ' + str(np.round(df['Means2'][i], decimals=2) ) + '+-'+ str(np.round(df['Means2err'][i], decimals=2) ),
            r'$\mu_3=$ ' + str(np.round(df['Means3'][i], decimals=2) ) + '+-'+ str(np.round(df['Means3err'][i], decimals=2) ),
            r'$\sigma=$ ' + str(np.round(df['Sigmas'][i], decimals = 2) ) + '+-'+ str(np.round(df['Sigmaserr'][i], decimals = 2) ),
            #r'$\sigma_2=$ ' + str(np.round(df['Sigmas2'][i], decimals = 2) ) + '+-'+ str(np.round(df['Sigmas2err'][i], decimals = 2) ),
            r'$\sigma_3=$ ' + str(np.round(df['Sigmas3'][i], decimals = 2) ) + '+-'+ str(np.round(df['Sigmas3err'][i], decimals = 2) ),
            r'$A=$ ' + str(np.format_float_scientific(df['A'][i], precision = 2) ) + '+-'+ str(np.format_float_scientific(df['Aerr'][i], precision = 2) ),
            r'$c_1=$ ' + str(np.format_float_scientific(df['c1'][i], precision = 2) ) + '+-'+ str(np.format_float_scientific(df['c1err'][i], precision = 2) ),
            r'$c_2=$ ' + str(np.format_float_scientific(df['c2'][i], precision = 2) ) + '+-'+ str(np.format_float_scientific(df['c2err'][i], precision = 2) ),
            r'$c_3=$ ' + str(np.format_float_scientific(df['c3'][i], precision = 2) ) + '+-'+ str(np.format_float_scientific(df['c3err'][i], precision = 2) )
            ))
                
                ax.text(0.03, 0.95, text, transform=ax.transAxes, fontsize=10, verticalalignment='top', bbox=props)
                
                ax.hist(df['Histogram Data'][i], bins=df['Bin Num'][i], histtype='step')
                ax.plot(df['X-axis'][i], df['Fit'][i], label='Fitted PDF', color='#000272')
                ax.plot(df['X-axis'][i], df['Guass1'][i], label='Primary Gaussian', linestyle='--', color = '#A72693')
                ax.plot(df['X-axis'][i], df['Guass2'][i], label='Secondary Gaussian', linestyle='--', color='#02383C')
                ax.plot(df['X-axis'][i], df['Guass3'][i], label='Third Gaussian', linestyle='--', color='#BD512F')
                ax.plot(df['X-axis'][i], df['linear'][i], label='Linear Bckg', linestyle='--', color='#615DEC')
                ax.set_title(df["Topologies"][i]+ ' (0.5 keV bins)')
                ax.legend(loc='upper right')
                
                ax2 = divider.append_axes("bottom", size="30%", pad=0.05)
                ax.figure.add_axes(ax2)
                ax2.errorbar(self.Residual_Plot(df['Histogram Data'][i], fitarr[i], bin_num[i])[0],self.Residual_Plot(df['Histogram Data'][i], fitarr[i], bin_num[i])[1], fmt='o',markersize=4)
                ax2.axhline(0, linestyle='--', color='crimson')
                
                
                
        plt.tight_layout()
        plt.show()
        
        return df
    
    def Double_GaussianMinimizer(self,data, title, initial_parms, bin_num):
        c = cost.UnbinnedNLL(data, self.Double_Gaussian)

        E0 = min(data)
        E1 = max(data)
        Ec = (E0+E1)/2
        bound_A = 1/(Ec-E0)

        m = Minuit(c,*initial_parms)
        m.limits["f", "d"] = (0, 1)
        m.limits["mu", 'mu2', 'sigma'] = (0, None)
        m.limits["A"] = (-bound_A, bound_A)
        m.migrad()
        m.hesse()

        x_values = np.linspace(min(data), max(data), 1000)
        bin_width = (max(data) - min(data))/bin_num

        fig = plt.figure()
        ax=fig.add_axes((.1,.3,.8,.6))
        
        props = dict(boxstyle='round', facecolor='whitesmoke', alpha=0.5)
        text = '\n'.join(( r'$\Delta\chi^2/ndf= $' + str(np.round(self.Chi2(data, m.values, bin_num), decimals=2)) +'/' +str(bin_num - (self.Double_Gaussian.__code__.co_argcount -1), ),
                        r'$P(ndf,\chi^2)=%.2f$' % (self.PValue(bin_num - (self.Double_Gaussian.__code__.co_argcount -1),self.Chi2(data, m.values, bin_num)), ),
            'Entries: ' + (str(len(data))),
            r'$\mu=$ ' + str(np.round(m.values[0], decimals = 2) ) + '+-'+ str(np.round(m.errors[0], decimals = 2) ),
            r'$\mu_2=$ ' + str(np.round(m.values[1], decimals=2) ) + '+-'+ str(np.round(m.errors[1], decimals=2) ),
            r'$\sigma=$ ' + str(np.round(m.values[2], decimals = 2) ) + '+-'+ str(np.round(m.errors[2], decimals = 2) ),
            #r'$\sigma_2=$ ' + str(np.round(m.values[3], decimals = 2) ) + '+-'+ str(np.round(m.errors[3], decimals = 2) ),
            r'$A=$ ' + str(np.format_float_scientific(m.values[3], precision = 2) ) + '+-'+ str(np.format_float_scientific(m.errors[3], precision = 2) ),
            r'$f=$ ' + str(np.format_float_scientific(m.values[4], precision = 2) ) + '+-'+ str(np.format_float_scientific(m.errors[4], precision = 2) ),
            r'$d=$ ' + str(np.format_float_scientific(m.values[5], precision = 2) ) + '+-'+ str(np.format_float_scientific(m.errors[5], precision = 2) )
            ))
        
        mu, mu2,sigma, A,f,d = m.values
        muerr, mu2err,sigmaerr, Aerr,ferr,derr = m.errors

        ax.hist(data, bins = bin_num , histtype='step');
        ax.plot(x_values, self.Double_Gaussian(x_values,*m.values)*len(data)*bin_width,label='Fitted PDF', color='#000272')
        ax.plot(x_values, (1-f-d)*norm.pdf(x_values,mu, sigma)*len(data)*bin_width, label='Primary Gaussian', linestyle='--', color='#A72693')
        ax.plot(x_values, f*(1+A*(x_values-Ec))/(E1-E0)*len(data)*bin_width, label='Linear Bckg', linestyle='--', color='#BD512F')
        ax.plot(x_values, d*norm.pdf(x_values,mu2, sigma)*len(data)*bin_width, label='Secondary Gaussian', linestyle='--', color='#02383C')
        ax.legend(loc='upper right')
        ax.text(0.03, 0.95, text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
        ax.set_title(title)

        ax2=fig.add_axes((.1,.08,.8,.2))        
        ax2.errorbar(self.Residual_Plot(data, m.values, bin_num)[0],self.Residual_Plot(data, m.values, bin_num)[1], fmt='o',markersize=4)
        ax2.axhline(0, linestyle='--', color='crimson')
        
        return mu, muerr, sigma, sigmaerr, self.Chi2(data, m.values, bin_num)
    
    def Double_GaussianMinimizerTopologies(self,data, topologies, initial_parms):
        fitarr = []
        errorarr = []
        PDFs = []
        guass1 = []
        guass2 = []
        linear = []
        xvals = []
        bin_num = []
        bin_width = []

        for i in range(len(data)):
            E0 = min(data[i])
            E1 = max(data[i])
            Ec = (E0+E1)/2
            bound_A = 1/(Ec-E0)
            c = cost.UnbinnedNLL(data[i], self.Double_Gaussian)
            m = Minuit(c,*initial_parms[i])
            m.limits["f", "d"] = (0, 1)
            m.limits["mu", 'mu2', 'sigma'] = (0, None)
            m.limits["A"] = (-bound_A, bound_A)
            m.migrad()
            m.hesse()
            
            fitarr.append(m.values)
            errorarr.append(m.errors)
            x_values = np.linspace(min(data[i]), max(data[i]), 1000)
            xvals.append(x_values)
            bin_num.append(2*int(E1-E0))
            
            bin_width.append((E1-E0)/(2*int(E1-E0)))

        chisq = []
        pvals = []
        means = []
        means2 = []
        sigmas = []
        # sigmas2 = []
        A = []
        f = []
        d = []

        meanserr = []
        means2err = []
        sigmaserr = []
        # sigmas2err = []
        Aerr= []
        ferr = []
        derr = []
            
        for i in range(len(fitarr)):
            E0 = min(data[i])
            E1 = max(data[i])
            Ec = (E0+E1)/2
            PDFs.append(self.Double_Gaussian(xvals[i],*fitarr[i])*len(data[i])*bin_width[i])
            guass1.append((1-fitarr[i][4]-fitarr[i][5])*norm.pdf(xvals[i],fitarr[i][0], fitarr[i][2])*len(data[i])*bin_width[i])
            guass2.append((fitarr[i][5])*norm.pdf(xvals[i],fitarr[i][1], fitarr[i][2])*len(data[i])*bin_width[i])
            linear.append(fitarr[i][4]*(1+fitarr[i][3]*(xvals[i]-Ec))/(E1-E0)*len(data[i])*bin_width[i])
            chisq.append(self.Chi2(data[i], fitarr[i], bin_num[i]))
            pvals.append(self.PValue(bin_num[i]-(self.Double_Gaussian.__code__.co_argcount -1), chisq[i]))
            
            means.append(fitarr[i][0])
            means2.append(fitarr[i][1])
            sigmas.append(fitarr[i][2])
            # sigmas2.append(fitarr[i][3])
            A.append(fitarr[i][3])
            f.append(fitarr[i][4])
            d.append(fitarr[i][5])
            
            meanserr.append(errorarr[i][0])
            means2err.append(errorarr[i][1])
            sigmaserr.append(errorarr[i][2])
            # sigmas2err.append(errorarr[i][3])
            Aerr.append(errorarr[i][3])
            ferr.append(errorarr[i][4])
            derr.append(errorarr[i][5])
            
            
            
        data = { 'Topologies': topologies,
            'Bin Num': bin_num,
        'Histogram Data': data,
            'X-axis': xvals,
            'Fit': PDFs,
            'Guass1': guass1,
            'Guass2': guass2,
            'Linear': linear,
            'Means': means,
            'Means2': means2,
            'Sigmas': sigmas,
            # 'Sigmas2': sigmas2,
            'A': A,
            'f': f,
            'd': d, 
            'Chi2': chisq,
            'p-value': pvals, 
            'Meanserr': meanserr,
            'Means2err': means2err,
            'Sigmaserr': sigmaserr,
            # 'Sigmas2err': sigmas2err,
            'Aerr': Aerr,
            'ferr': ferr,
            'derr': derr,}

        df = pd.DataFrame(data)
            
        fig, axs = plt.subplots(2, 2, figsize=(10, 8))
        props = dict(boxstyle='round', facecolor='whitesmoke')
        for i, ax in enumerate(axs.flatten()):
            if i < len(df):
                divider = make_axes_locatable(ax)
                
                text = '\n'.join(( r'$\Delta\chi^2/ndf= $' + str(np.round(df['Chi2'][i], decimals=2)) +'/' +str(df['Bin Num'][i] - (Double_Gaussian.__code__.co_argcount -1), ),
                        r'$P(ndf,\chi^2)=%.2f$' % (df['p-value'][i], ),
            'Entries: ' + (str(len(df['Histogram Data'][i]))),
            r'$\mu=$ ' + str(np.round(df['Means'][i], decimals = 2) ) + '+-'+ str(np.round(df['Meanserr'][i], decimals = 2) ),
            r'$\mu_2=$ ' + str(np.round(df['Means2'][i], decimals=2) ) + '+-'+ str(np.round(df['Means2err'][i], decimals=2) ),
            r'$\sigma=$ ' + str(np.round(df['Sigmas'][i], decimals = 2) ) + '+-'+ str(np.round(df['Sigmaserr'][i], decimals = 2) ),
            # r'$\sigma_2=$ ' + str(np.round(df['Sigmas2'][i], decimals = 2) ) + '+-'+ str(np.round(df['Sigmas2err'][i], decimals = 2) ),
            r'$A=$ ' + str(np.format_float_scientific(df['A'][i], precision = 2) ) + '+-'+ str(np.format_float_scientific(df['Aerr'][i], precision = 2) ),
            r'$f=$ ' + str(np.format_float_scientific(df['f'][i], precision = 2) ) + '+-'+ str(np.format_float_scientific(df['ferr'][i], precision = 2) ),
            r'$d=$ ' + str(np.format_float_scientific(df['d'][i], precision = 2) ) + '+-'+ str(np.format_float_scientific(df['derr'][i], precision = 2) )
            ))
                
                ax.text(0.03, 0.95, text, transform=ax.transAxes, fontsize=10, verticalalignment='top', bbox=props)
                
                ax.hist(df['Histogram Data'][i], bins=df['Bin Num'][i], histtype='step')
                ax.plot(df['X-axis'][i], df['Fit'][i], label='Fitted PDF', color='#000272')
                ax.plot(df['X-axis'][i], df['Guass1'][i], label='Primary Gaussian', linestyle='--', color='#A72693')
                ax.plot(df['X-axis'][i], df['Guass2'][i], label='Secondary Gaussian', linestyle='--', color='#02383C')
                ax.plot(df['X-axis'][i], df['Linear'][i], label='Linear Background', linestyle='--', color='#BD512F')
                ax.set_title(df["Topologies"][i] + ' (0.5 keV bins)')
                ax.legend(loc='upper right')
                
                ax2 = divider.append_axes("bottom", size="30%", pad=0.05)
                ax.figure.add_axes(ax2)
                ax2.errorbar(self.Residual_Plot(df['Histogram Data'][i], fitarr[i], bin_num[i])[0],self.Residual_Plot(df['Histogram Data'][i], fitarr[i], bin_num[i])[1], fmt='o',markersize=4)
                ax2.axhline(0, linestyle='--', color='crimson')
                
        plt.tight_layout()
        plt.show()
        
        return df['Means'], df['Meanserr'], df['Sigmas'], df['Sigmaserr'], df['Chi2']
    
    def TripleGaussianMinimizer(self,data, initial_parms, title, bin_num):
        c = cost.UnbinnedNLL(data, self.TripleGaussian)

        m = Minuit(c, *initial_parms)
        m.limits["c2", 'c3'] = (0, 1)
        m.limits["mu1", 'mu2', 'mu3', "sigma"] = (0, None)

        m.migrad()
        m.hesse()

        x_values = np.linspace(min(data), max(data), 1000)
        bin_width = (max(data) - min(data))/bin_num

        fig = plt.figure()
        ax=fig.add_axes((.1,.3,.8,.6))
        
        props = dict(boxstyle='round', facecolor='whitesmoke', alpha=0.5)
        text = '\n'.join(( r'$\Delta\chi^2/ndf= $' + str(np.round(self.Chi2(data, m.values, bin_num), decimals=2)) +'/' +str(bin_num - (self.TripleGaussian.__code__.co_argcount -1), ),
                        r'$P(ndf,\chi^2)=%.2f$' % (self.PValue(bin_num - (self.TripleGaussian.__code__.co_argcount -1),self.Chi2(data, m.values, bin_num)), ),
            'Entries: ' + (str(len(data))),
            r'$\mu_1=$ ' + str(np.round(m.values[0], decimals = 2) ) + '+-'+ str(np.round(m.errors[0], decimals = 2) ),
            r'$\mu_2=$ ' + str(np.round(m.values[1], decimals=2) ) + '+-'+ str(np.round(m.errors[1], decimals=2) ),
            r'$\mu_3=$ ' + str(np.round(m.values[2], decimals = 2) ) + '+-'+ str(np.round(m.errors[2], decimals = 2) ),
            r'$\sigma=$ ' + str(np.round(m.values[3], decimals = 2) ) + '+-'+ str(np.round(m.errors[3], decimals = 2) ),
            r'$c_2=$ ' + str(np.format_float_scientific(m.values[4], precision = 2) ) + '+-'+ str(np.format_float_scientific(m.errors[4], precision = 2) ),
            r'$c_3=$ ' + str(np.format_float_scientific(m.values[5], precision = 2) ) + '+-'+ str(np.format_float_scientific(m.errors[5], precision = 2) )
            ))
            
        mu1,mu2,mu3, sigma, c2,c3 = m.values

        ax.hist(data, bins = bin_num , histtype="step");
        ax.plot(x_values, self.TripleGaussian(x_values,*m.values)*len(data)*bin_width,label='Fitted PDF',color='#000272')
        ax.plot(x_values, (1-c2-c3)*norm.pdf(x_values,mu1,sigma)*len(data)*bin_width, label='Gaussian 1', linestyle='--', color = '#BD512F')
        ax.plot(x_values, c2*norm.pdf(x_values,mu2,sigma)*len(data)*bin_width, label='Gaussian 2', linestyle='--', color='#02383C')
        ax.plot(x_values, c3*norm.pdf(x_values,mu3,sigma)*len(data)*bin_width, label='Gaussian 3', linestyle='--',color='#615DEC')
        ax.text(0.03, 0.95, text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
        ax.legend()

        ax2=fig.add_axes((.1,.08,.8,.2))        
        ax2.errorbar(self.Residual_Plot(data, m.values, bin_num)[0],self.Residual_Plot(data, m.values, bin_num)[1], fmt='o',markersize=4)
        ax2.axhline(0, linestyle='--', color='crimson')
        ax.set_title(title)
        
        return mu1, m.errors[0], sigma, m.errors[3], self.Chi2(data, m.values, bin_num)
    
    def TripleGaussianMinimizerTopologies(self, data, topologies, initial_parms, wid, height, figwidth, figheight):
        fitarr = []
        errorarr = []
        PDFs = []
        guass1 = []
        guass2 = []
        guass3 = []
        xvals = []
        bin_num = []
        bin_width = []

        for i in range(len(data)):
            E0 = min(data[i])
            E1 = max(data[i])
            c = cost.UnbinnedNLL(data[i], self.TripleGaussian)
            m = Minuit(c,*initial_parms[i])
            m.limits["c2", "c3"] = (0, 1)
            m.limits["mu1", 'mu2', 'sigma'] = (0, None)
            m.migrad()
            m.hesse()
            
            fitarr.append(m.values)
            errorarr.append(m.errors)
            x_values = np.linspace(min(data[i]), max(data[i]), 1000)
            xvals.append(x_values)
            bin_num.append(int(E1-E0))
            
            bin_width.append((E1-E0)/(int(E1-E0)))

        chisq = []
        pvals = []
        means1 = []
        means2 = []
        means3 = []
        sigmas = []
        c2 = []
        c3 = []

        means1err = []
        means2err = []
        means3err = []
        sigmaserr = []
        c2err= []
        c3err = []
            
        for i in range(len(fitarr)):
            E0 = min(data[i])
            E1 = max(data[i])
            PDFs.append(self.TripleGaussian(xvals[i],*fitarr[i])*len(data[i])*bin_width[i])
            guass1.append((1-fitarr[i][4]-fitarr[i][5])*norm.pdf(xvals[i],fitarr[i][0], fitarr[i][3])*len(data[i])*bin_width[i])
            guass2.append((fitarr[i][4])*norm.pdf(xvals[i],fitarr[i][1], fitarr[i][3])*len(data[i])*bin_width[i])
            guass3.append(fitarr[i][5]*norm.pdf(xvals[i], fitarr[i][2], fitarr[i][3])*len(data[i])*bin_width[i])
            chisq.append(self.Chi2(data[i], fitarr[i], bin_num[i]))
            pvals.append(self.PValue(bin_num[i]-(self.TripleGaussian.__code__.co_argcount -1), chisq[i]))
            
            means1.append(fitarr[i][0])
            means2.append(fitarr[i][1])
            means3.append(fitarr[i][2])
            sigmas.append(fitarr[i][3])
            c2.append(fitarr[i][4])
            c3.append(fitarr[i][5])
            
            means1err.append(errorarr[i][0])
            means2err.append(errorarr[i][1])
            means3err.append(errorarr[i][2])
            sigmaserr.append(errorarr[i][3])
            c2err.append(errorarr[i][4])
            c3err.append(errorarr[i][5])
            
            
            
        data = { 'Topologies': topologies,
            'Bin Num': bin_num,
        'Histogram Data': data,
            'X-axis': xvals,
            'Fit': PDFs,
            'Guass1': guass1,
            'Guass2': guass2,
            'Guass3': guass3,
            'means1': means1,
            'means2': means2,
            'means3': means3,
            'Sigmas': sigmas,
            'c2': c2,
            'c3': c3, 
            'Chi2': chisq,
            'p-value': pvals, 
            'means1err': means1err,
            'means2err': means2err,
            'means3err': means3err,
            'Sigmaserr': sigmaserr,
            'c2err': c2err,
            'c3err': c3err}

        df = pd.DataFrame(data)
            
        fig, axs = plt.subplots(wid, height, figsize=(figwidth, figheight))
        props = dict(boxstyle='round', facecolor='whitesmoke')
        for i, ax in enumerate(axs.flatten()):
            if i < len(df):
                divider = make_axes_locatable(ax)
                
                text = '\n'.join(( r'$\Delta\chi^2/ndf= $' + str(np.round(df['Chi2'][i], decimals=2)) +'/' +str(df['Bin Num'][i] - (self.TripleGaussian.__code__.co_argcount -1), ),
                        r'$P(ndf,\chi^2)=%.2f$' % (df['p-value'][i], ),
            'Entries: ' + (str(len(df['Histogram Data'][i]))),
            r'$\mu=$ ' + str(np.round(df['means1'][i], decimals = 2) ) + '+-'+ str(np.round(df['means1err'][i], decimals = 2) ),
            r'$\mu_2=$ ' + str(np.round(df['means2'][i], decimals=2) ) + '+-'+ str(np.round(df['means2err'][i], decimals=2) ),
            r'$\mu_3=$ ' + str(np.round(df['means3'][i], decimals=2) ) + '+-'+ str(np.round(df['means3err'][i], decimals=2) ),
            r'$\sigma=$ ' + str(np.round(df['Sigmas'][i], decimals = 2) ) + '+-'+ str(np.round(df['Sigmaserr'][i], decimals = 2) ),
            r'$c_2=$ ' + str(np.format_float_scientific(df['c2'][i], precision = 2) ) + '+-'+ str(np.format_float_scientific(df['c2err'][i], precision = 2) ),
            r'$c_3=$ ' + str(np.format_float_scientific(df['c3'][i], precision = 2) ) + '+-'+ str(np.format_float_scientific(df['c3err'][i], precision = 2) )
            ))
                
                ax.text(0.03, 0.95, text, transform=ax.transAxes, fontsize=7, verticalalignment='top', bbox=props)
                
                ax.hist(df['Histogram Data'][i], bins=df['Bin Num'][i], histtype="step")
                ax.plot(df['X-axis'][i], df['Fit'][i], label='Fitted PDF', color='#000272')
                ax.plot(df['X-axis'][i], df['Guass1'][i], label='Gaussian 1', linestyle='--', color = '#BD512F')
                ax.plot(df['X-axis'][i], df['Guass2'][i], label='Gaussian 2', linestyle='--', color='#02383C')
                ax.plot(df['X-axis'][i], df['Guass3'][i], label='Gaussian 3', linestyle='--', color='#615DEC')
                ax.set_title(df["Topologies"][i])
                ax.legend(loc='upper right')
                
                ax2 = divider.append_axes("bottom", size="30%", pad=0.05)
                ax.figure.add_axes(ax2)
                ax2.errorbar(self.Residual_Plot(df['Histogram Data'][i], fitarr[i], bin_num[i])[0],self.Residual_Plot(df['Histogram Data'][i], fitarr[i], bin_num[i])[1], fmt='o',markersize=4)
                ax2.axhline(0, linestyle='--', color='crimson')
                
        plt.tight_layout()
        plt.show()
        
        return df
    
    def NormGaussianExp(self, x,mu,sigma,b,C):
        E0 = min(x)
        E1 = max(x)
        return (1-b)*(norm.pdf(x,mu,sigma))+b*(np.exp(-C*x))/((1/C)*(np.exp(-E0*C)-np.exp(-E1*C)))
    
    def NormGuassianExpMinimizer(self, data, title, initial_parms, bin_num):
        c = cost.UnbinnedNLL(data, self.NormGaussianExp)

        E0 = min(data)
        E1 = max(data)
        Ec = (E0+E1)/2
        bound_A = 1/(Ec-E0)

        m = Minuit(c,*initial_parms)
        m.limits["b"] = (0, 1)
        m.limits["mu", 'sigma'] = (0, None)
        m.migrad()
        m.hesse()

        x_values = np.linspace(min(data), max(data), 1000)
        bin_width = (max(data) - min(data))/bin_num

        fig = plt.figure()
        ax=fig.add_axes((.1,.3,.8,.6))
        
        props = dict(boxstyle='round', facecolor='whitesmoke', alpha=0.5)
        text = '\n'.join(( r'$\Delta\chi^2/ndf= $' + str(np.round(self.Chi2(data, m.values, bin_num), decimals=2)) +'/' +str(bin_num - (self.NormGaussianExp.__code__.co_argcount -1), ),
                        r'$P(ndf,\chi^2)=%.2f$' % (self.PValue(bin_num - (self.NormGaussianExp.__code__.co_argcount -1),self.Chi2(data, m.values, bin_num)), ),
            'Entries: ' + (str(len(data))),
            r'$\mu=$ ' + str(np.round(m.values[0], decimals = 2) ) + '+-'+ str(np.round(m.errors[0], decimals = 2) ),
            r'$\sigma=$ ' + str(np.round(m.values[1], decimals = 2) ) + '+-'+ str(np.round(m.errors[1], decimals = 2) ),
            r'$b=$ ' + str(np.format_float_scientific(m.values[2], precision = 2) ) + '+-'+ str(np.format_float_scientific(m.errors[2], precision = 2) ),
            r'$C=$ ' + str(np.format_float_scientific(m.values[3], precision = 2) ) + '+-'+ str(np.format_float_scientific(m.errors[3], precision = 2) )
            ))
        
        mu, sigma, b, C = m.values
        muerr, sigmaerr, berr,Cerr = m.errors

        ax.hist(data, bins = bin_num , histtype="step",fill=False);
        ax.plot(x_values,self.NormGaussianExp(x_values,*m.values)*len(data)*bin_width,label='Fitted PDF', color='#02383C')
        ax.plot(x_values, (1-b)*norm.pdf(x_values,mu, sigma)*len(data)*bin_width, label='Primary Gaussian', linestyle='--', color='#BD512F')
        ax.plot(x_values, b*(np.exp(-C*x_values))/((1/C)*(np.exp(-E0*C)-np.exp(-E1*C)))*len(data)*bin_width, label='Exp Bckg', linestyle='--', color='#615DEC')
        ax.legend(loc='upper right')
        ax.text(0.03, 0.95, text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
        ax.set_title(title)

        ax2=fig.add_axes((.1,.08,.8,.2))        
        ax2.errorbar(self.Residual_Plot(data, m.values, bin_num)[0],self.Residual_Plot(data, m.values, bin_num)[1], fmt='o',markersize=4)
        ax2.axhline(0, linestyle='--', color='crimson')
        
        return mu, muerr, sigma, sigmaerr, self.Chi2(data, m.values, bin_num)
    
    def NormGaussianExpMinimizerTopology(self,data, topologies, initial_parms, wid, height, figwidth, figheight):
        fitarr = []
        errorarr = []
        PDFs = []
        gauss = []
        Exp = []
        xvals = []
        bin_num = []
        bin_width = []

        for i in range(len(data)):
            E0 = min(data[i])
            E1 = max(data[i])
            c = cost.UnbinnedNLL(data[i], self.NormGaussianExp)
            m = Minuit(c,*initial_parms[i])
            m.limits["b"] = (0, 1)
            m.limits["mu",'sigma'] = (0, None)
            m.migrad()
            m.hesse()
            
            fitarr.append(m.values)
            errorarr.append(m.errors)
            x_values = np.linspace(min(data[i]), max(data[i]), 1000)
            xvals.append(x_values)
            bin_num.append(int(E1-E0))
            
            bin_width.append((E1-E0)/(int(E1-E0)))

        chisq = []
        pvals = []
        means = []
        sigmas = []
        b = []
        C = []

        meanserr = []
        sigmaserr = []
        berr = []
        Cerr = []
            
        for i in range(len(fitarr)):
            E0 = min(data[i])
            E1 = max(data[i])
            PDFs.append(self.NormGaussianExp(xvals[i],*fitarr[i])*len(data[i])*bin_width[i])
            gauss.append((1-fitarr[i][2])*norm.pdf(xvals[i],fitarr[i][0], fitarr[i][1])*len(data[i])*bin_width[i])
            Exp.append(fitarr[i][2]*(np.exp(-fitarr[i][3]*xvals[i]))/((1/fitarr[i][3])*(np.exp(-E0*fitarr[i][3])-np.exp(-E1*fitarr[i][3])))*len(data[i])*bin_width[i])
            chisq.append(self.Chi2(data[i], fitarr[i], bin_num[i]))
            pvals.append(self.PValue(bin_num[i]-(self.NormGaussianExp.__code__.co_argcount -1), chisq[i]))
            
            means.append(fitarr[i][0])
            sigmas.append(fitarr[i][1])
            b.append(fitarr[i][2])
            C.append(fitarr[i][3])
            
            meanserr.append(errorarr[i][0])
            sigmaserr.append(errorarr[i][1])
            berr.append(errorarr[i][2])
            Cerr.append(errorarr[i][3])
            
        data = { 'Topologies': topologies,
            'Bin Num': bin_num,
        'Histogram Data': data,
            'X-axis': xvals,
            'Fit': PDFs,
            'gauss': gauss,
            'Exp': Exp,
            'Means': means,
            'Sigmas': sigmas,
            'b': b,
            'C': C, 
            'Chi2': chisq,
            'p-value': pvals, 
            'Meanserr': meanserr,
            'Sigmaserr': sigmaserr,
            'berr': berr,
            'Cerr': Cerr,}

        df = pd.DataFrame(data)
            
        fig, axs = plt.subplots(wid, height, figsize=(figwidth, figheight))
        props = dict(boxstyle='round', facecolor='whitesmoke')
        for i, ax in enumerate(axs.flatten()):
            if i < len(df):
                divider = make_axes_locatable(ax)
                
                text = '\n'.join(( r'$\Delta\chi^2/ndf= $' + str(np.round(df['Chi2'][i], decimals=2)) +'/' +str(df['Bin Num'][i] - (self.NormGaussianExp.__code__.co_argcount -1), ),
                        r'$P(ndf,\chi^2)=%.2f$' % (df['p-value'][i], ),
            'Entries: ' + (str(len(df['Histogram Data'][i]))),
            r'$\mu=$ ' + str(np.round(df['Means'][i], decimals = 2) ) + '+-'+ str(np.round(df['Meanserr'][i], decimals = 2) ),
            r'$\sigma=$ ' + str(np.round(df['Sigmas'][i], decimals = 2) ) + '+-'+ str(np.round(df['Sigmaserr'][i], decimals = 2) ),
            r'$b=$ ' + str(np.format_float_scientific(df['b'][i], precision = 2) ) + '+-'+ str(np.format_float_scientific(df['berr'][i], precision = 2) ),
            r'$C=$ ' + str(np.format_float_scientific(df['C'][i], precision = 2) ) + '+-'+ str(np.format_float_scientific(df['Cerr'][i], precision = 2) )
            ))
                
                ax.text(0.03, 0.95, text, transform=ax.transAxes, fontsize=8, verticalalignment='top', bbox=props)
                
                ax.hist(df['Histogram Data'][i], bins=df['Bin Num'][i], histtype="step")
                ax.plot(df['X-axis'][i], df['Fit'][i], label='Fitted PDF', color='#02383C')
                ax.plot(df['X-axis'][i], df['gauss'][i], label='Gaussian', linestyle='--', color='#BD512F')
                ax.plot(df['X-axis'][i], df['Exp'][i], label='Exp Background', linestyle='--', color='#615DEC')
                ax.set_title(df["Topologies"][i])
                ax.legend(loc='upper right')
                
                ax2 = divider.append_axes("bottom", size="30%", pad=0.05)
                ax.figure.add_axes(ax2)
                ax2.errorbar(self.Residual_Plot(df['Histogram Data'][i], fitarr[i], bin_num[i])[0],self.Residual_Plot(df['Histogram Data'][i], fitarr[i], bin_num[i])[1], fmt='o',markersize=4)
                ax2.axhline(0, linestyle='--', color='crimson')
                
        plt.tight_layout()
        plt.show()
        
        return df
    
    def ExpectedVals(self, total_dataframe, dataframe):
        weights = []
        for dataset_id in range(3801, 3829):
            val = (((len(total_dataframe[total_dataframe['Dataset']==dataset_id]))/len(total_dataframe))*100)/(((len(dataframe[dataframe['Dataset']==dataset_id]))/len(dataframe))*100)
            weights.append(val)
            
        df_bias = pd.read_csv(self.bias, sep=',', names=['Dataset', 'Bias', 'Bias_err'], header=None)
        df_res = pd.read_csv(self.res, sep=',', names=['Dataset', 'Res', 'Res_err'], header=None)
        
        mean_bias = 0
        sigma_res = 0
        
        for i in range(len(weights)):
                mean_bias += weights[i]*df_bias.iloc[i+1,1]
                sigma_res += weights[i]*df_res.iloc[i+1,1]
                
        average_means_bias = mean_bias/sum(weights)
        average_sigma_res = sigma_res/sum(weights)
        
        mean_bias_err = 0
        sigma_res_err = 0
        
        for i in range(len(weights)):
                mean_bias_err += (weights[i]*df_bias.iloc[i+1,2])**2
                sigma_res_err += (weights[i]*df_res.iloc[i+1,2])**2
        
        actual_mean_err = 1/sum(weights)*np.sqrt(mean_bias_err)
        actual_sigma_err = 1/sum(weights)*np.sqrt(sigma_res_err)
        
        
        return average_means_bias, (average_sigma_res/2.355), actual_mean_err, actual_sigma_err*(1/2.355)
            
    

        
