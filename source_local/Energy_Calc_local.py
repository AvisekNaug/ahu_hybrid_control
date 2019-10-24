# import modules
import numpy as np
from source.ThermoProp_local import *
from joblib import load


def KtoF(tempK):
    tempF = (tempK-273.15)*9/5 + 32
    return tempF


def FtoK(tempF):
    tempK = 273.15+(tempF - 32)*5/9
    return tempK


def cfm2m3(cfm):
    return cfm/35.315


class energy_calc():

    def __init__(self, modelpath: list):
        self.precoolmodel = load(modelpath[0])
        self.recovheatmodel = load(modelpath[1])

    def preheatenergy(self, airflow, oat, ph_temp, orh):

        Tdb_in_ph = FtoK(oat)  # K
        Tdb_out_ph = FtoK(ph_temp)  # K
        W_in_ph = HumidityRatio(Tdb_in_ph, orh)  # returns kg-of-moisture/kg-DryAir
        v = specificVolume(Tdb_in_ph, W_in_ph)  # m3/kg
        m_a = cfm2m3(airflow)/v  # kg/s
        H_in = Enthalpy_Air_H2O(Tdb_in_ph, W_in_ph)  # kJ/kg
        W_out_ph = W_in_ph  # Sensible heating
        H_out = Enthalpy_Air_H2O(Tdb_out_ph, W_out_ph)  # kJ/kg
        pht_energy = m_a*(H_out - H_in)  # energy consumed in kJ/s
        pht_out_rh = RelativeHumidty(W_out_ph, Tdb_out_ph)  # unitless relative humidity

        # if we follow current controller settings
        ph_temp_hist = 75  # F
        Tdb_out_ph = FtoK(ph_temp_hist)  # K
        H_out = Enthalpy_Air_H2O(Tdb_out_ph, W_out_ph)  # kJ/kg
        pht_energy_hist = m_a * (H_out - H_in)  # energy consumed in kJ/s
        pht_out_rh_hist = RelativeHumidty(W_out_ph, Tdb_out_ph)  # unitless relative humidity

        return pht_energy, pht_out_rh, pht_energy_hist, pht_out_rh_hist

    def precooltemp(self, ph_temp, airflow, pht_out_rh, pht_out_rh_hist):

        precooltemp = self.precoolmodel.predict(np.array([ph_temp, airflow]).reshape(1, -1))  # F
        W_in = HumidityRatio(FtoK(ph_temp), pht_out_rh)  # returns kg-of-moisture/kg-DryAir
        W_out=W_in  # sensible cooling
        pc_out_rh = RelativeHumidty(W_out, FtoK(precooltemp))  # unit less relative humidity

        # if we follow current controller settings
        ph_temp_hist = 75  # F
        precooltemp_hist = self.precoolmodel.predict(np.array([ph_temp_hist, airflow]).reshape(1, -1))  # F
        W_in_hist = HumidityRatio(FtoK(ph_temp_hist), pht_out_rh_hist)  # returns kg-of-moisture/kg-DryAir
        W_out_hist = W_in_hist  # sensible cooling
        pc_out_rh_hist = RelativeHumidty(W_out_hist, FtoK(precooltemp_hist))  # unitless relative humidity

        return precooltemp, pc_out_rh, precooltemp_hist, pc_out_rh_hist

    def coolingenergy(self, precooltemperature, cc_t, airflow, pc_out_rh,
                      pc_out_rh_hist, precooltemp_hist):
        T_db_in_cc = FtoK(precooltemperature)
        Tdb_out_cc = FtoK(cc_t)  # K
        W_in_cc = HumidityRatio(T_db_in_cc, pc_out_rh)  # returns kg-of-moisture/kg-DryAir
        v = specificVolume(T_db_in_cc, W_in_cc)  # m3/kg
        m_a = cfm2m3(airflow) / v  # kg/s
        H_in = Enthalpy_Air_H2O(T_db_in_cc, W_in_cc)
        H_w = specificEnthalpyWater(Tdb_out_cc)  # Latent heat of cooling
        W_out_cc = HumidityRatio(Tdb_out_cc, 1.0)  # returns Kg-of-moisture/Kg-DryAir RH_out=1.0
        H_out = Enthalpy_Air_H2O(Tdb_out_cc, W_out_cc)
        cc_energy = m_a*(H_in - H_out) + m_a*(W_out_cc - W_in_cc)*H_w
        cc_out_rh = 1.0

        # if we follow current controller settings
        T_db_in_cc = FtoK(precooltemp_hist)
        W_in_cc = HumidityRatio(T_db_in_cc, pc_out_rh_hist)  # returns kg-of-moisture/kg-DryAir
        v = specificVolume(T_db_in_cc, W_in_cc)  # m3/kg
        m_a = cfm2m3(airflow) / v  # kg/s
        H_in = Enthalpy_Air_H2O(T_db_in_cc, W_in_cc)
        cc_energy_hist = m_a * (H_in - H_out) + m_a * (W_out_cc - W_in_cc) * H_w
        cc_out_rh_hist = 1.0

        return cc_energy, cc_out_rh, cc_energy_hist, cc_out_rh_hist

    def recovheattemp(self, oat, ph_temp, cc_t, airflow, cc_out_rh, cc_out_rh_hist):

        recovtemp = self.recovheatmodel.predict(np.array([oat, ph_temp, cc_t, airflow]).reshape(1, -1))
        W_in = HumidityRatio(FtoK(cc_t), cc_out_rh)  # returns kg-of-moisture/kg-DryAir
        W_out = W_in  # sensible heating
        recov_out_rh = RelativeHumidty(W_out, FtoK(recovtemp))  # unit less relative humidity

        # if we follow current controller settings
        ph_temp_hist = FtoK(75)  # K
        recovtemp_hist = self.recovheatmodel.predict(np.array([oat, ph_temp_hist, cc_t, airflow]).reshape(1, -1))
        W_in = HumidityRatio(FtoK(cc_t), cc_out_rh_hist)  # returns kg-of-moisture/kg-DryAir
        W_out = W_in  # sensible heating
        recov_out_rh_hist = RelativeHumidty(W_out, FtoK(recovtemp_hist))  # unit less relative humidity

        return recovtemp, recov_out_rh, recovtemp_hist, recov_out_rh_hist

    def reheatenergy(self, airflow, recovtemp, recov_out_rh,
                     recovtemp_hist, recov_out_rh_hist, rht_stp, rht_stp_hist):

        if recovtemp < rht_stp:
            T_rh_in = FtoK(recovtemp)
            T_rh_out = FtoK(rht_stp)
            W_in_rht = HumidityRatio(T_rh_in, recov_out_rh)
            v = specificVolume(T_rh_in, W_in_rht)  # m3/kg
            m = cfm2m3(airflow)/v
            H_in = Enthalpy_Air_H2O(T_rh_in, W_in_rht)
            W_out_rht = W_in_rht  # Sensible heating
            H_out = Enthalpy_Air_H2O(T_rh_out, W_out_rht)
            rht_energy = m*(H_out-H_in)
            rht_out_rh = RelativeHumidty(W_out_rht, T_rh_out)
            sat = rht_stp
        else:
            rht_energy = 0
            rht_out_rh = recov_out_rh
            sat = recovtemp

        # if we follow current controller settings
        if recovtemp_hist < rht_stp_hist:
            T_rh_in = FtoK(recovtemp_hist)
            T_rh_out = FtoK(rht_stp_hist)
            W_in_rht = HumidityRatio(T_rh_in, recov_out_rh_hist)
            v = specificVolume(T_rh_in, W_in_rht)  # m3/kg
            m = cfm2m3(airflow) / v
            H_in = Enthalpy_Air_H2O(T_rh_in, W_in_rht)
            W_out_rht = W_in_rht  # Sensible heating
            H_out = Enthalpy_Air_H2O(T_rh_out, W_out_rht)
            rht_energy_hist = m * (H_out - H_in)
            rht_out_rh_hist = RelativeHumidty(W_out_rht, T_rh_out)
            sat_hist = rht_stp_hist
        else:
            rht_energy_hist = 0
            rht_out_rh_hist = recov_out_rh_hist
            sat_hist = recovtemp_hist

        return rht_energy, rht_out_rh, sat, rht_energy_hist, rht_out_rh_hist, sat_hist
