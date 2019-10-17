#Importing my own packages which may contain some bugs
import ThermoProp
import importlib
import numpy as np
#need to run this if I make changes to Energy_Calc and other modules
importlib.reload(ThermoProp)
from ThermoProp import *
from joblib import dump, load

def KtoF(tempK):
    tempF = (tempK-273.15)*9/5 + 32
    return tempF

def FtoK(tempF):
    tempK = 273.15+(tempF - 32)*5/9
    return tempK

#Calculating the cooling energy
class Energy_Calc():
    def __init__(self):
        #Contact Factor for the heating coil
        self.StDev = 1.2876331552182259
        self.RecovHeatModel = load('HeatRecov2.joblib')
        #Format for prediction: learner.predict(np.array([[1,2,3,4]]))
        self.RecovCoolModel = load('CoolRecov.joblib')
        
    def updateVars(self,rowInfo):#Updates the relevant properties of the air vapor mixture
        self.ccoutput = rowInfo[3]
        self.rhoutput = rowInfo[4]
        
        self.P = rowInfo[14]*3386.389 # to Pa
        self.airflow = rowInfo[2]*0.0004719474 # to m3/s
        
        self.Tdb_in_cc = 273.15+(rowInfo['recovCoolTemp'] - 32)*5/9 #K
        self.Tdb_in_cc_F = rowInfo['recovCoolTemp']#F
        self.recovHeatedTemp = rowInfo['recovHeatTemp']

        self.dewPT_at_CC = 273.15+(rowInfo[13] - 32)*5/9 #K
        #Since preheat coil does sensible heating and recov coil does sensible cooling, dewPT is the same
        self.ccSPT = 273.15+(rowInfo[7] - 32)*5/9 #K
        self.CC_T= 273.15 + (rowInfo[10] - 32)*5/9 #K
        self.Tdb_out_rht= 273.15 + (rowInfo[11] - 32)*5/9 #K        
        self.Tdb_in_pht = 273.15+(rowInfo[8] - 32)*5/9 #K ##!! can alos use rowInfo[0] since prht isnt operating        
        self.RH_in = rowInfo[12]/100#fraction
        
        self.W_in_cc = HumidityRatio(self.Tdb_in_pht,self.RH_in,self.P)# returns Kg-of-moisture/Kg-DryAir
             
    def RecovHeat(self,newCC_T):
        #model predicts 1st order difference
        recovHeatTemp = self.RecovHeatModel.predict(np.array([[self.recovHeatedTemp,KtoF(self.Tdb_in_pht),KtoF(self.CC_T)]]))[0] + newCC_T
        return recovHeatTemp
    
    def CooledAirTemp(self,CC_STP):
        newCC_T = CC_STP + np.random.normal(0,self.StDev,1)[0]
        #assuming that its value at t+1 is a function of m_a1,pht/oat,cc_t; otherwise just shift this line to 
        #updateVars method
        recovHeatTemp = self.RecovHeat(newCC_T)#Passing updated CC_T as we are directly calculating the 
        #recovHeatTemp at next state
        
        return newCC_T,recovHeatTemp
        
    def CC_in(self):
        
        return cc_in
        
    def CoolingEnergy(self,T_cc_org):
        if self.ccoutput>0:#encodes the humidity changeover setpoint information?
            #Since pht and recov cool is sensible heating and cooling respectively, W_in_pht=W_in_cc remains same
            v_1 = specificVolume(self.Tdb_in_cc,self.W_in_cc,self.P/100)#m3/Kg
            m_a1 = self.airflow/v_1#Kg/s
            H_in = Enthalpy_Air_H2O(self.Tdb_in_cc, self.W_in_cc)
            
            if self.dewPT_at_CC>=self.ccSPT:# ie always ends at saturation line with 100% RH and CC_T

                H_w = specificEnthalpyWater(self.CC_T)#Latent heat of cooling
                W_out_cc = HumidityRatio(self.CC_T,1.0,self.P)# returns Kg-of-moisture/Kg-DryAir RH_out=1.0
                H_out = Enthalpy_Air_H2O(self.CC_T, W_out_cc)
                Energy = m_a1*(H_in-H_out)-m_a1*(self.W_in_cc-W_out_cc)*H_w
                
                Energy_org = self.orgCooling(T_cc_org,H_in,m_a1)
                                
                return Energy,Energy_org

            else:# dewPT_at_CC < ccSPT ie only sensible cooling is done,
                W_out_cc = self.W_in_cc
                H_out = Enthalpy_Air_H2O(self.CC_T, W_out_cc)
                Energy = m_a1*(H_in-H_out)
                
                Energy_org = self.orgCooling(T_cc_org,H_in,m_a1)
                
                return Energy,Energy_org
        else:
            return 0
        
    def orgCooling(self,T_cc_org,H_in,m_a1):
        
        if self.dewPT_at_CC>=T_cc_org:# ie always ends at saturation line with 100% RH and CC_T
            H_w_org = specificEnthalpyWater(T_cc_org)#Latent heat of cooling
            W_out_cc_org = HumidityRatio(T_cc_org,1.0,self.P)# returns Kg-of-moisture/Kg-DryAir RH_out=1.0
            H_out_org = Enthalpy_Air_H2O(T_cc_org, W_out_cc_org)
            Energy_org = m_a1*(H_in-H_out_org)-m_a1*(self.W_in_cc-W_out_cc_org)*H_w_org
            
        else:# dewPT_at_CC < T_cc_org ie only sensible cooling is done,
            W_out_cc_org = self.W_in_cc
            H_out_org = Enthalpy_Air_H2O(T_cc_org, W_out_cc_org)
            Energy_org = m_a1*(H_in-H_out_org)
            
        return Energy_org
            
        
    #Calculating the reheating energy
    def ReheatEnergy(self,T_con_out):
        #Either air is at 100% RH and ActualCooledTemp or it is at W_in_cc humidity ratio with ActualCooledTemp
        if self.rhoutput>0:
        
            if self.dewPT_at_CC>=self.ccSPT:# ie always starts at saturation line with 100% RH and CCT_T

                Tdb_in_rht = 273.15 + (self.recovHeatedTemp - 32)*5/9 #K
                RH_in = 1.0 #fraction: RH leaving the cooling coil
                W_in_rht = HumidityRatio(self.CC_T,RH_in,self.P)# returns Kg-of-moisture/Kg-DryAir
                v_2 = specificVolume(Tdb_in_rht,W_in_rht,self.P/100)#m3/Kg
                m_a2 = self.airflow/v_2

                H_in = Enthalpy_Air_H2O(Tdb_in_rht, W_in_rht)
                W_out_rht=W_in_rht #Sensible heating
                H_out = Enthalpy_Air_H2O(self.Tdb_out_rht,W_out_rht)
                
                #Calculate RH if building controller setpoint is followed
                T_con_out = FtoK(T_con_out) #F to K
                H_out_org = Enthalpy_Air_H2O(T_con_out,W_out_rht)
                
                #Calculating this as it is needed for reward function
                self.RH_rheat_out = RelativeHumidty(W_out_rht,self.Tdb_out_rht,101325)
                self.RH_rheat_org = RelativeHumidty(W_out_rht,T_con_out,101325)

                Energy = m_a2*(H_out-H_in)
                Energy_org = m_a2*(H_out_org-H_in)
                return Energy,Energy_org

            else:#at W_in_cc humidity ratio with CC_T
                Tdb_in_rht = 273.15 + (self.recovHeatedTemp - 32)*5/9 #K
                v_2 = specificVolume(Tdb_in_rht,self.W_in_cc,self.P/100)#m3/Kg
                m_a2 = airflow/v_2

                W_in_rht=self.W_in_cc
                W_out_rht=self.W_in_cc #Sensible heating

                H_in = Enthalpy_Air_H2O(Tdb_in_rht, W_in_rht)
                H_out = Enthalpy_Air_H2O(self.Tdb_out_rht,W_out_rht)
                
                #Calculate RH if building controller setpoint is followed
                T_con_out = 273.15+(T_con_out - 32)*5/9 #F to K
                H_out_org = Enthalpy_Air_H2O(T_con_out,W_out_rht)
                
                #Calculating this as it is needed for reward function
                self.RH_rheat_out = RelativeHumidty(W_out_rht,self.Tdb_out_rht,101325)
                self.RH_rheat_org = RelativeHumidty(W_out_rht,T_con_out,101325)

                Energy = m_a2*(H_out-H_in)
                Energy_org = m_a2*(H_out_org-H_in)
                return Energy,Energy_org

        else:
            return 0
        #totalData['Reheat_energy'] = totalData.apply(lambda x: ReheatEnergy(x) , axis=1)

    def preheatenergy(self,airflow,oat,pht_stp):
        pass
        