package io.github.dominikthomas.neuralnet.beans;

import java.util.ArrayList;
import io.github.dominikthomas.neuralnet.init.UniformInitialization;
import io.github.dominikthomas.neuralnet.init.WeightInitialization;
import io.github.dominikthomas.neuralnet.math.IActivationFunction;

/**
 *
 * NeuralNet
 * This class represents the Neural Network itself. It contains all the 
 * definitions that a Neural Network has, including method for calculation 
 * (forward).
 * 
 * @author Alan de Souza, Fábio Soares, Dominik Thomas
 * @version 0.4
 */
public class NeuralNet {
    
    /**
     * Neural Network Input Layer
     */
    private InputLayer inputLayer;
    /**
     * Neural Network array of hidden layers, that may contain 0 or many
     */
    private ArrayList<HiddenLayer> hiddenLayers;
    /**
     * Neural Network Output Layer
     */
    private OutputLayer outputLayer;
    /**
     * Output Activation function
     */
    private IActivationFunction outputAcFnc;
    /**
     * Number of Hidden Layers
     */
    private int numberOfHiddenLayers = 0;
    /**
     * Number of Inputs
     */
    private int numberOfInputs;
    /**
     * Number of Outputs
     */
    private int numberOfOutputs;
    
    /**
     * Array of neural inputs
     */
    private ArrayList<Double> input;
    /**
     * Array of neural outputs
     */
    private ArrayList<Double> output;
    
    private boolean activeBias=true;
    
    private WeightInitialization weightInitialization
    =new UniformInitialization(0.0,1.0);
    
    public enum NeuralNetMode { BUILD, TRAINING, RUN };
    
    private NeuralNetMode neuralNetMode = NeuralNetMode.BUILD;
    
    /**
     * NeuralNet constructor
     * This constructor initializes the neural network by initializing all of 
     * the underlying layers and their respective neurons.
     * 
     * @param numberofinputs Number of Inputs of this Neural Network
     * @param numberofoutputs Number of Outputs of this Neural Network
     * @param numberofhiddenneurons Array containing the number of Neurons in 
     * each of the Hidden Layers
     * @param hiddenAcFnc Array containing the activation function of each 
     * Hidden Layer
     * @param outputAcFnc Activation Function of the Output Layer
     */
    public NeuralNet(){
    }
    
    public void init(){
    	if(hiddenLayers != null) { 
    		numberOfHiddenLayers=hiddenLayers.size();
        }
        input=new ArrayList<>(numberOfInputs);
        inputLayer=new InputLayer(numberOfInputs);
        if(numberOfHiddenLayers>0){
	        for(int i=0;i<numberOfHiddenLayers;i++){
	            if(i==0){
                    hiddenLayers.get(i).setPreviousLayer(inputLayer);
                    hiddenLayers.get(i).setNumberOfInputs(inputLayer.getNumberOfNeuronsInLayer());
                    inputLayer.setNextLayer(hiddenLayers.get(i));
	            } else{
                    hiddenLayers.get(i).setPreviousLayer(hiddenLayers.get(i-1));
                    hiddenLayers.get(i).setNumberOfInputs(hiddenLayers.get(i-1).getNumberOfNeuronsInLayer());
                    hiddenLayers.get(i-1).setNextLayer(hiddenLayers.get(i));
                }
	            hiddenLayers.get(i).init();
            }
            outputLayer=new OutputLayer(numberOfOutputs,outputAcFnc,
                    hiddenLayers.get(numberOfHiddenLayers-1)
                    .getNumberOfNeuronsInLayer() 
                    );
            hiddenLayers.get(numberOfHiddenLayers-1).setNextLayer(outputLayer);
        } else {
            outputLayer=new OutputLayer(numberOfOutputs, outputAcFnc,
            		numberOfInputs);
            inputLayer.setNextLayer(outputLayer);
        }
        if(activeBias) {
        	activateBias();
        } else {
        	deactivateBias();
        }
    }
    
    /**
     * setInputs
     * Feeds an array of real values to the neural network's inputs
     * @param inputs Array of real values to be fed into the neural inputs
     */
    public void setInputs(ArrayList<Double> inputs){
        if(inputs.size()==numberOfInputs){
            this.input=inputs;
        }
    }
    
    /**
     * setInputs
     * Sets a vector of double-precision values into the neural network inputs
     * @param inputs vector of values to be fed into the neural inputs
     * @throws NeuralException 
     */
    public void setInputs(double[] inputs) throws NeuralException{
        if(inputs.length==numberOfInputs){
            for(int i=0;i<numberOfInputs;i++){
                try{
                    input.set(i, inputs[i]);
                }
                catch(IndexOutOfBoundsException iobe){
                    input.add(inputs[i]);
                }
            }
        } else {
        	throw new NeuralException("Number of inputs is different from the value of the numberOfInputs variable.");
        }
    }
    
    /**
     * calc
     * This method calculates the output of each layer and forwards all values 
     * to the next layer
     */
    public void calc(){
        inputLayer.setInputs(input);
        inputLayer.calc();
        if(numberOfHiddenLayers>0){
            for(int i=0;i<numberOfHiddenLayers;i++){
                HiddenLayer hl = hiddenLayers.get(i);
                hl.init();
                hl.setInputs(hl.getPreviousLayer().getOutputs());
                hl.calc();
            }
        }
        outputLayer.setInputs(outputLayer.getPreviousLayer().getOutputs());
        outputLayer.calc();
        this.output=outputLayer.getOutputs();
    }
    
    /**
     * getArrayOutputs
     * @return Returns the neural outputs in the form of Array
     */
    public ArrayList<Double> getArrayOutputs(){
        return output;
    }
    
    /**
     * getOutputs
     * @return Returns the neural outputs in the form of vector
     */
    public double[] getOutputs(){
        double[] _outputs = new double[numberOfOutputs];
        for(int i=0;i<numberOfOutputs;i++){
            _outputs[i]=output.get(i);
        }
        return _outputs;
    }
    
    /**
     * print
     * Method to print the neural network information
     */
    public void print(){
        System.out.println("Neural Network: "+this.toString());
        System.out.println("\tInputs:"+String.valueOf(this.numberOfInputs));
        System.out.println("\tOutputs:"+String.valueOf(this.numberOfOutputs));
        System.out.println("\tHidden Layers: "+String.valueOf(numberOfHiddenLayers));
        for(int i=0;i<numberOfHiddenLayers;i++){
            System.out.println("\t\tHidden Layer "+
                    String.valueOf(i)+": "+
                    String.valueOf(this.hiddenLayers.get(i)
                            .numberOfNeuronsInLayer)+" Neurons");
        }
        
    }
    
    public void deactivateBias(){
        if(numberOfHiddenLayers>0){
            for(HiddenLayer hl:hiddenLayers){
                for(Neuron n:hl.getListOfNeurons()){
                    n.deactivateBias();
                }
            }
        }
        for(Neuron n:outputLayer.getListOfNeurons()){
            n.deactivateBias();
        }
    }
    
    public void activateBias(){
    	if(numberOfHiddenLayers>0){
	        for(HiddenLayer hl:hiddenLayers){
	            for(Neuron n:hl.getListOfNeurons()){
	                n.activateBias();
	            }
	        }
    	}
        for(Neuron n:outputLayer.getListOfNeurons()){
            n.activateBias();
        }
    }
    
    public boolean isBiasActive(){
        return activeBias;
    }
    
    public void setActiveBias(boolean activeBias) {
		this.activeBias = activeBias;
	}
    
    public HiddenLayer getHiddenLayer(int i){
        return hiddenLayers.get(i);
    }
    
    public OutputLayer getOutputLayer(){
        return outputLayer;
    }
    
    public int getNumberOfInputs(){
        return numberOfInputs;
    }
    
    public int getNumberOfHiddenLayers(){
        return numberOfHiddenLayers;
    }
    
    public int getNumberOfOutputs(){
        return numberOfOutputs;
    }
    
    public Double getInput(int i){
        return input.get(i);
    }
    
    public double[] getInputs(){
        double[] result=new double[numberOfInputs];
        for(int i=0;i<numberOfInputs;i++){
            result[i]=input.get(i);
        }
        return result;
    }
    
    public WeightInitialization getWeightInitialization(){
        return weightInitialization;
    }
    
    public void setNumberOfInputs(int numberOfInputs) {
		this.numberOfInputs = numberOfInputs;
	}
    
    public void setNumberOfOutputs(int numberOfOutputs) {
		this.numberOfOutputs = numberOfOutputs;
	}
    
    public void setOutputAcFnc(IActivationFunction outputAcFnc) {
		this.outputAcFnc = outputAcFnc;
	}
    
    public void setHiddenLayers(ArrayList<HiddenLayer> hiddenLayers) {
		this.hiddenLayers = hiddenLayers;
	}

    public void setNeuralNetMode(NeuralNetMode neuralNetMode) {
		this.neuralNetMode = neuralNetMode;
	}
    
    public void setWeightInitialization(WeightInitialization weightInitialization) {
		this.weightInitialization = weightInitialization;
	}
    
    
}
