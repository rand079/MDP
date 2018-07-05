/* Code used for 'Predicting concept drift in data streams using metadata clustering' as presented at IJCNN '18
 * 
 * This code is made available for research reproducability. For any other purposes, please contact the author first at rand079 at aucklanduni dot ac dot nz
 * This code is not in a refined state. Please contact the author if anything does not seem to work as expected.
 */

package IJCNNMDPGit;

import java.io.OutputStream;
import java.io.PrintStream;
import java.util.Arrays;
import java.util.LinkedList;
import org.apache.commons.math3.util.FastMath;

import com.yahoo.labs.samoa.instances.Attribute;
import com.yahoo.labs.samoa.instances.DenseInstance;
import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.Instances;
import com.yahoo.labs.samoa.instances.InstancesHeader;

import moa.AbstractMOAObject;
import moa.cluster.CFCluster;
import moa.cluster.Cluster;
import moa.cluster.Clustering;
import moa.clusterers.denstream.MicroCluster;
import moa.clusterers.denstream.WithDBSCAN;
import moa.clusterers.macro.NonConvexCluster;
import moa.core.AutoExpandVector;
import moa.core.FastVector;

public class MetadataDriftPredictor extends AbstractMOAObject{

	private static final long serialVersionUID = 1L;
	public WithDBSCAN driftPredictor;
	InstancesHeader driftInstanceHeader;
	
	//Metadata to use
	boolean useAccuracy = true;
	boolean useVolatility = true;
	boolean useRSev = true;
	boolean usePSev = true;
	boolean warningDetection;
	
	//When mode = "Sensitive", returned drift chance will always be >= 0.5. When "Conservative", will always be <= 0.5.
	//Otherwise, defaults to predictive where returned 0 <= drift chance <= 1.0.
	String mode; 
	
	Clustering driftClusters = null;
	
	static double severityWindowLength = 30; //window to measure severity metadata over
	public static int bufferSize; //number of drifts before starting to adapt i.e. dmin parameter
	int driftsSeen = 0; //count of drifts seen by MDP instance
	
	//Stream information
	int numClasses;
	int numAttributes;
	Instances buffer = new Instances();
	
	int offlineClusteringInterval;  //frequency of rebuilding clustering, defaults to dmin
	double microClusterRadiusMultiplier = 1.0;
	
	double currentResponseSeverity = 0.0;
	double currentPredictorSeverity = 0.0;
	int instSinceDrift = 0;
	public int countSkips = 0;
	
	double oneOverSqrtTwo = 1/(FastMath.sqrt(2));
	
	double minAcc = 2;
	double maxAcc = -1;
	double[] varMinimums;
	double[] varMaximums;
	LinkedList<Double> recentResponseVars = new LinkedList<Double>();
	double[] responseVarsAfterDrift;
	LinkedList<double[]> recentPredictorVars = new LinkedList<double[]>();;
	double[] predictorVarsAfterDrift;	
	
	private PrintStream realSystemOut = System.out;
	private static class NullOutputStream extends OutputStream {
	    @Override
	    public void write(int b){
	         return;
	    }
	    @Override
	    public void write(byte[] b){
	         return;
	    }
	    @Override
	    public void write(byte[] b, int off, int len){
	         return;
	    }
	    public NullOutputStream(){
	    }
	}
	
	//Parameters: drifts till clustering (dmin), num of classes in data, num of atts in data, "Sensitive"/"Conservative"/"Other", 
	//array of 5 booleans values to determine metadata set
	public MetadataDriftPredictor(int bufferSizeNum, int numOfClasses, int numOfAttributes,String adaptmode, boolean[] metadataToUse) {
		useAccuracy = metadataToUse[0];
		useVolatility = metadataToUse[1];
		useRSev = metadataToUse[2];
		usePSev = metadataToUse[3];
		warningDetection = metadataToUse[4];
		driftPredictor = new WithDBSCAN();
		this.bufferSize = bufferSizeNum;
		driftPredictor.initPointsOption.setValue(bufferSize);
		int numMetadataColumns = (useAccuracy?1:0)+(useVolatility?1:0)+(useRSev?1:0)+(usePSev?1:0)+(warningDetection?1:0);
		driftPredictor.prepareForUse();
		generateDriftInstanceHeader();
		numClasses = numOfClasses;
		numAttributes = numOfAttributes;
		mode = adaptmode;
		int offlineClusteringInterval = bufferSizeNum; 
		if(mode.equals("Sensitive")) microClusterRadiusMultiplier = 1.5;
		if(mode.equals("Conservative")) microClusterRadiusMultiplier = 0.5;
		
		if (warningDetection){
			varMinimums = new double[5];
			varMaximums = new double[5];
		}else{
			varMinimums = new double[4];
			varMaximums = new double[4];
		}
		Arrays.fill(varMinimums, Double.MAX_VALUE);
		Arrays.fill(varMaximums, Double.MIN_VALUE);
		
		recentResponseVars = new LinkedList();
		recentPredictorVars = new LinkedList();
		responseVarsAfterDrift = null;
		predictorVarsAfterDrift = null;
	}
	
	//Predict how close we are to drifting based on current stream state
	public double getDriftPrediction(double currAcc, int interval, int warningLength){
		Instance inst = createInstance(currAcc, interval, warningLength, false);
		if(!(driftClusters == null) && driftClusters.size()>0){
	        double driftChance;
	        inst = normalizeInstance(inst);
	        driftChance = getInclusionProb(inst);
		 	if(mode.equals("Sensitive")){
		 			if(driftChance < 0.5) driftChance = 0.5;
		 	} else if(mode.equals("Conservative")){
				if(driftChance > 0.5) driftChance = 0.5;
			}
			return driftChance;
			
		} else {
        	return 0.5;	
        }    
	}
	
	//Pass every stream instance to drift predictor to get response and predictor distance
	public void learnDataDist(Instance inst){
		instSinceDrift++;
		recentResponseVars.add(inst.classValue());
		if(recentResponseVars.size() == severityWindowLength && responseVarsAfterDrift == null){
			responseVarsAfterDrift = new double[numClasses];
			int i = 0;
			Arrays.fill(responseVarsAfterDrift, 0);
			for(double d:recentResponseVars){
				this.responseVarsAfterDrift[(int)d]++;
			}
		}
		if(recentResponseVars.size() > severityWindowLength) {
			recentResponseVars.removeFirst();
			computeResponseSeverity();
		}
	
		recentPredictorVars.add(inst.toDoubleArray());
		if(recentPredictorVars.size() == severityWindowLength && predictorVarsAfterDrift == null){
			predictorVarsAfterDrift = new double[numAttributes];
			Arrays.fill(predictorVarsAfterDrift, 0);
			
			for(double d[]:recentPredictorVars){
				for(int i = 0; i < numAttributes - 1; i++)
					predictorVarsAfterDrift[i]+= d[i];
			}
			
			for(int i = 0; i < predictorVarsAfterDrift.length; i++)
				predictorVarsAfterDrift[i] = predictorVarsAfterDrift[i]/severityWindowLength;
		}
		if(recentPredictorVars.size() > severityWindowLength) {
			recentPredictorVars.removeFirst();
			computePredictorSeverity();
		}
	}
	
	//Learn what a drift state looks like
	public void learnDriftState(double currAcc, int interval, int warningLength){
		
		if(driftsSeen == 0) buffer = new Instances(this.driftInstanceHeader);
		
		driftsSeen++;
		Instance inst = createInstance(currAcc, interval, warningLength, true);
		
		if(driftsSeen <= bufferSize) buffer.add(inst);
		if(driftsSeen == bufferSize) buildClustering();
		if(driftsSeen >= bufferSize){
			driftPredictor.trainOnInstance(normalizeInstance(inst));
			System.setOut(new PrintStream(new NullOutputStream()));
			if(driftsSeen % offlineClusteringInterval == 0)
				driftClusters = driftPredictor.getClusteringResult();
			System.setOut(realSystemOut);
		}
		
		recentResponseVars = new LinkedList();
		recentPredictorVars = new LinkedList();
		responseVarsAfterDrift = null;
		predictorVarsAfterDrift = null;
		currentResponseSeverity = 0.0;
		currentPredictorSeverity = 0.0;
		minAcc = 2;
		maxAcc = -1;
		instSinceDrift = 0;
	}

	//Create metadata for current stream
	public Instance createInstance(double currAcc, int interval, int warningLength, boolean isDrift){
		
		Instance inst = new DenseInstance(this.driftInstanceHeader.numAttributes());

		double logAcc = logPlusN(currAcc, 1);
		double logInt = logPlusN((double)interval,1);
		double logRSev = logPlusN(currentResponseSeverity, 1);
		double logPSev = logPlusN(currentPredictorSeverity, 1);
		double logWarn = logPlusN((double)warningLength, 1);

		int index = 0;
		inst.setDataset(this.driftInstanceHeader);
        if(useAccuracy){
        	inst.setValue(index, logAcc);
        	index++;
        }
        if(useVolatility){
        	inst.setValue(index, logInt);
	        index++;
        }
        if (useRSev){
            inst.setValue(index, logRSev);
            index++;
        }
        if(usePSev){
            inst.setValue(index, logPSev);
            index++;
        }
        if(warningDetection){
	        inst.setValue(index, logWarn);
	        index++;
        }
        
        if((driftsSeen < bufferSize & isDrift) | (bufferSize == 0 & isDrift)){
        	if(logAcc < varMinimums[0]) varMinimums[0] = logAcc;
        	if(logAcc > varMaximums[0]) varMaximums[0] = logAcc;
        	if(logInt < varMinimums[1]) varMinimums[1] = logInt;
        	if(logInt > varMaximums[1]) varMaximums[1] = logInt;
        	if(logRSev < varMinimums[2]) varMinimums[2] = logRSev;
        	if(logRSev > varMaximums[2]) varMaximums[2] = logRSev;
        	if(logPSev < varMinimums[3]) varMinimums[3] = logPSev;
        	if(logPSev > varMaximums[3]) varMaximums[3] = logPSev;
            
            if(warningDetection){
            	if(logWarn < varMinimums[4]) varMinimums[4] = logWarn;
            	if(logWarn > varMaximums[4]) varMaximums[4] = logWarn;
            }
        }
        return inst;
        
	}
	
	private void generateDriftInstanceHeader(){
		FastVector attributes = new FastVector();
        if (useAccuracy) attributes.addElement(new Attribute("acc"));
        if (useVolatility) attributes.addElement(new Attribute("interval"));
        if (useRSev) attributes.addElement(new Attribute("Rsev"));
        if (usePSev) attributes.addElement(new Attribute("Psev"));
        if (warningDetection) attributes.addElement(new Attribute("warningInterval"));

        InstancesHeader driftHeader = new InstancesHeader(
        		new Instances("drift", attributes, 0));
        this.driftInstanceHeader = driftHeader;
	}
	
	//Use Hellinger distance to measure difference between class distribution after drift point and now using window
	public void computeResponseSeverity() {
		int[] responseVarsCurrent = new int[numClasses];
		Arrays.fill(responseVarsCurrent, 0);
		for(double d:recentResponseVars){
			responseVarsCurrent[(int)d]++;
		}
		double hellingerDist = 0.0;
		for(int i = 0; i < numClasses; i++){
			hellingerDist += FastMath.pow(FastMath.sqrt((double)responseVarsAfterDrift[i]/severityWindowLength) -
					FastMath.sqrt((double)responseVarsCurrent[i]/(double)severityWindowLength),2);
		}
		//System.out.println(Math.sqrt(hellingerDist));
		this.currentResponseSeverity =  FastMath.sqrt(hellingerDist);
		if(Double.isNaN(this.currentPredictorSeverity))
			System.out.print("Issue with response severity");
	}
	
	//Use scaled Hellinger distance to measure difference between mean attribute values after drift point and now using window
	public void computePredictorSeverity() {
		double[] predictorVarsCurrent = new double[numAttributes];
		Arrays.fill(predictorVarsCurrent, 0);
		
		for(double d[]:recentPredictorVars){
			for(int i = 0; i < numAttributes - 1; i++)
				predictorVarsCurrent[i]+= d[i];
		}
		
		for(int i = 0; i < predictorVarsCurrent.length; i++)
			predictorVarsCurrent[i] = predictorVarsCurrent[i]/severityWindowLength;
		
		double hellingerDist = 0.0;
		for(int i = 0; i < numAttributes; i++){
			double a = (double)predictorVarsAfterDrift[i];
			double b = (double)predictorVarsCurrent[i];
			if((a<0) != (b<0)){
				if(a < b){
					b = b + Math.abs(a);
					a = 0;
				} else {
					a = a + Math.abs(b);
					b = 0;
				}
			}
			if(a != 0 & b!=0){
				double this_max = Math.max(a,b);
				a = a/this_max;
				b = b/this_max;
			}

			
			hellingerDist += FastMath.pow(FastMath.sqrt(a/severityWindowLength) -
					FastMath.sqrt(b/(double)severityWindowLength),2);
		}
		this.currentPredictorSeverity =  FastMath.sqrt(hellingerDist);
		if(Double.isNaN(this.currentPredictorSeverity))
			System.out.print("Issue with predictor severity");
	}
	
	//Normalize metadata by maximums seen so far in stream
	public Instance normalizeInstance(Instance inst){
		for(int a = 0; a < inst.numAttributes(); a++){
			inst.setValue(a, (inst.value(a)- varMinimums[a])/(varMaximums[a] - varMinimums[a]));
		}
		return inst;
	}
	
	public void buildClustering(){
		//normalize data and build clustering with first bufferSize instances
		for(int i = 0; i < bufferSize; i++){
			Instance inst = buffer.get(i);
			inst = normalizeInstance(inst);
			driftPredictor.trainOnInstance(inst);
		}
		
		System.out.println("Clustering built");
	}
	
	public double logPlusN(double num, double plus){
		return FastMath.log(num+plus);
	}
	
	//Custom function for getting prob in [0,1] for cluster membership from Clustering
	public double getInclusionProb(Instance inst)  {
        double maxInclusion = 0.0;
        AutoExpandVector<Cluster> clusters = driftClusters.getClustering();
        for (int i = 0; i < clusters.size(); i++) {
            maxInclusion = Math.max(getClusterInclusionProbability((NonConvexCluster)clusters.get(i), inst),
                    maxInclusion);
        }
        return maxInclusion;
    }
	
	//Custom function for getting prob in [0,1] for cluster membership from a NonConvexCluster made of microclusters
	public double getClusterInclusionProbability(NonConvexCluster cluster, Instance inst) {
		double maxInclusion = 0;
		for (CFCluster cf : cluster.getMicroClusters()) {
            maxInclusion = Math.max(getMicroClusterInclusionProbability((MicroCluster)cf, inst),
                    maxInclusion);
		}
		return maxInclusion;
	}

	//Custom function for getting prob in [0,1] based on proximity to a given microcluster
	public double getMicroClusterInclusionProbability(MicroCluster mc, Instance inst) {
		double d = mc.getCenterDistance(inst);
		double rad = mc.getRadius()*microClusterRadiusMultiplier;
		if(d < rad)
			return 1- 1/(1+Math.exp(-5.0*(d/rad-1)));
		else return 1 - 1/(1+Math.exp(-1.0*(d/rad-1)));
    }

	@Override
	public void getDescription(StringBuilder sb, int indent) {
		// TODO Auto-generated method stub
		
	}
}
