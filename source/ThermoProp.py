import math
# Predicting the thermodynamic properties
from CoolProp.CoolProp import PropsSI,Props
from CoolProp.HumidAirProp import HAPropsSI

def Part_press(P,W):
    
    ''' Function to compute partial vapor pressure in [kPa]
        From page 6.9 equation 38 in ASHRAE Fundamentals handbook (2005)
            P = ambient pressure [kPa]
            W = humidity ratio [kg/kg dry air]
    '''
    result = P * W / (0.62198 + W)
    return result


def Sat_press(Tdb):

    ''' Function to compute saturation vapor pressure in [kPa]
        ASHRAE Fundamentals handbood (2005) p 6.2, equation 5 and 6
            Tdb = Dry bulb temperature [degC]
            Valid from -100C to 200 C
    '''

    C1 = -5674.5359
    C2 = 6.3925247
    C3 = -0.009677843
    C4 = 0.00000062215701
    C5 = 2.0747825E-09
    C6 = -9.484024E-13
    C7 = 4.1635019
    C8 = -5800.2206
    C9 = 1.3914993
    C10 = -0.048640239
    C11 = 0.000041764768
    C12 = -0.000000014452093
    C13 = 6.5459673
 
    TK = Tdb + 273.15                     # Converts from degC to degK
    
    if TK <= 273.15:
        result = math.exp(C1/TK + C2 + C3*TK + C4*TK**2 + C5*TK**3 + 
                          C6*TK**4 + C7*math.log(TK)) / 1000
    else:
        result = math.exp(C8/TK + C9 + C10*TK + C11*TK**2 + C12*TK**3 + 
                          C13*math.log(TK)) / 1000
    return result

def Hum_rat2(Tdb, RH, P=101.325):

    ''' Function to calculate humidity ratio [kg H2O/kg air]
        Given dry bulb and wet bulb temperature inputs [degC]
        ASHRAE Fundamentals handbood (2005)
            Tdb = Dry bulb temperature [degC]
            RH = Relative Humidity [Fraction or %]
            P = Ambient Pressure [kPa]
    '''
    Pws = Sat_press(Tdb)
    result = 0.62198*RH*Pws/(P - RH*Pws)    # Equation 22, 24, p6.8
    return result

def Rel_hum2(Tdb, W, P=101.325):
    
    ''' Calculates the relative humidity given:
            Tdb = Dry bulb temperature [degC]
            P = ambient pressure [kPa]
            W = humidity ratio [kg/kg dry air]
    '''

    Pw = Part_press(P, W)
    Pws = Sat_press(Tdb)
    result = Pw / Pws
    return result

#Functions which are preferred for calculating different factors
#H/h
def Enthalpy_Air_H2O(Tdb, W):#Enthalpy of dry air
    
    ''' Calculates enthalpy in kJ/kg (dry air) given:
            Tdb = Dry bulb temperature [K]
            W = Humidity Ratio [kg/kg dry air]
        Calculations from 2005 ASHRAE Handbook - Fundamentals - SI P6.9 eqn 32
    '''
    # C:\Users\nauga\Google Drive\BuildingProject\LiteratureReview\ModelEquations
    Tdb  = Tdb - 273.15
    result = 1.006*Tdb + W*(2501 + 1.86*Tdb)
    return result

#h_w
def specificEnthalpyWater(T):  # returns kJ/Kg of condensed water at T K
    return PropsSI('H', 'T', T, 'Q', 0, 'Water')/1000  # gives specific centhalpy of water which condensed at TK

#W
def HumidityRatio(Tdb,R,P=101325):  # returns Kg-of-moisture/Kg-DryAir at sea level eg HumidityRatio(293.15,0.6)
    return HAPropsSI('W','T',Tdb,'R',R ,'P',P)  # Kg moisture/Kg Dry Air

#R
def RelativeHumidty(W,Tdb,P=101325):  # returns relative humidity of air for W:kg/kg humidity ratio and T in Kelvin
    try:
        result = HAPropsSI('R','W',W,'T',Tdb,'P',P)  # unitless
    except ValueError:
        result = 1.0
    return result

#v_1/v
def specificVolume(Tdb,W,P=101.325):#Tdb:K; W=kg/kg; P=kPa
    return 0.2871*(Tdb)*(1 + 1.6078*W)/P#m3/Kg