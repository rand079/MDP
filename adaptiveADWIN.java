package IJCNNMDPGit;

import moa.classifiers.core.driftdetection.ADWIN;
import moa.classifiers.core.driftdetection.ADWINChangeDetector;

public class adaptiveADWIN extends ADWINChangeDetector implements AdaptiveDetector {
	
	double adaptRange = 2; //a parameter
	double adaptiveConf;
	
	public void setAdaptRange(double newRange){
		adaptRange = newRange;
	}
	
	public void adaptConf(double adaptation){
		//adaptation is prob of drift between 0-1 from drift predictor
		if(adaptation < 0.5){
			adaptiveConf = this.deltaAdwinOption.getValue() * (1/adaptRange + (2 * adaptation * (adaptRange-1)/adaptRange));
		}	
		else {
			adaptiveConf = this.deltaAdwinOption.getValue() * (1 + ((2 * (adaptation-0.5))*(adaptRange-1))); 
		}
	}
	
    @Override
    public void input(double inputValue) {
    	if(adaptiveConf <= 0) adaptiveConf = this.deltaAdwinOption.getValue();
        if (this.adwin == null) {
            resetLearning();
        }
        this.isChangeDetected = adwin.setInput(inputValue, adaptiveConf);
        this.isWarningZone = false;
        this.delay = 0.0;
        this.estimation = adwin.getEstimation();
    }
    
    @Override
    public void resetLearning() {
        adwin = new ADWIN((double) this.deltaAdwinOption.getValue());
        adaptRange = 2;
        adaptiveConf = this.deltaAdwinOption.getValue();
    }
	
}
	