package IJCNNMDPGit;

import org.apache.commons.math3.distribution.NormalDistribution;

import moa.classifiers.core.driftdetection.ADWINChangeDetector;
import moa.classifiers.core.driftdetection.DDM;
import moa.classifiers.core.driftdetection.HDDM_A_Test;

public class adaptiveDDM extends DDM implements AdaptiveDetector  {
		
	double adaptRange = 2; //a parameter
	double adaptConf = 3; //number of sd equivalent to 0.5 drift chance
	static NormalDistribution d = new NormalDistribution(0, 1);
	
    private int m_n;

    private double m_p;

    private double m_s;

    private double m_psmin;

    private double m_pmin;

    private double m_smin;
	  
    public double getZScore(double conf) { //get z-score for desired confidence
        return d.cumulativeProbability(conf);
    }
    
	public void setAdaptRange(double newRange){
		adaptRange = newRange;
	}
	
	public void adaptConf(double adaptation){
		// DDM uses 3 SDs i.e. 0.9987 prob which we adapt
		adaptConf = (2.6 + getZScore(1 - (1 + ((2.0 * (adaptation-0.5))*(adaptRange-1))) * (1 - 0.9955)))/2.0;
	}
    
    @Override
    public void resetLearning() {
        m_n = 1;
        m_p = 1;
        m_s = 0;
        m_psmin = Double.MAX_VALUE;
        m_pmin = Double.MAX_VALUE;
        m_smin = Double.MAX_VALUE;
        adaptConf = 3;
    }
	
    @Override
    public void input(double prediction) {
        // prediction must be 1 or 0
        // It monitors the error rate
        if (this.isChangeDetected == true || this.isInitialized == false) {
            resetLearning();
            this.isInitialized = true;
        }
        m_p = m_p + (prediction - m_p) / (double) m_n;
        m_s = Math.sqrt(m_p * (1 - m_p) / (double) m_n);

        m_n++;

        // System.out.print(prediction + " " + m_n + " " + (m_p+m_s) + " ");
        this.estimation = m_p;
        this.isChangeDetected = false;
        this.isWarningZone = false;
        this.delay = 0;

        if (m_n < this.minNumInstancesOption.getValue()) {
            return;
        }

        if (m_p + m_s <= m_psmin) {
            m_pmin = m_p;
            m_smin = m_s;
            m_psmin = m_p + m_s;
        }
        
        if (m_n > this.minNumInstancesOption.getValue() && m_p + m_s > m_pmin + adaptConf * m_smin) {
            //System.out.println(m_p + ",D");
            this.isChangeDetected = true;
            //resetLearning();
        } else if (m_p + m_s > m_pmin + 2 * m_smin) {
            //System.out.println(m_p + ",W");
            this.isWarningZone = true;
        } else {
            this.isWarningZone = false;
            //System.out.println(m_p + ",N");
        }
    }
	
}
	