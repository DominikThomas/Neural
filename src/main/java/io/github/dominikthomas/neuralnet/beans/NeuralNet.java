package io.github.dominikthomas.neuralnet.beans;

import io.github.dominikthomas.neuralnet.data.NeuralDataSet;
import io.github.dominikthomas.neuralnet.init.UniformInitialization;
import io.github.dominikthomas.neuralnet.init.WeightInitialization;

import java.util.ArrayList;

/**
 *
 * This class represents the Neural Network itself. It contains all the 
 * definitions that a Neural Network has, including method for calculation 
 * (forward).
 * 
 * @author Alan de Souza, Fabio Soares, Dominik Thomas
 * @version 0.2
 */
public class NeuralNet {
    
    /**
     * Neural Network Input Layer
     */
    protected InputLayer inputLayer;
    /**
     * Neural Network array of hidden layers, that may contain 0 or many
     */
    protected ArrayList<HiddenLayer> hiddenLayers;
    /**
     * Neural Network Output Layer
     */
    protected OutputLayer outputLayer;
    /**
     * Number of Hidden Layers
     */
    protected int numberOfHiddenLayers = 0;
    /**
     * Number of Inputs
     */
    protected int numberOfInputs;
    /**
     * Number of Outputs
     */
    protected int numberOfOutputs;
    
    /**
     * Array of neural inputs
     */
    protected ArrayList<Double> input;
    /**
     * Array of neural outputs
     */
    protected ArrayList<Double> output;
    
    
    protected boolean activeBias=true;
    
    
    protected WeightInitialization weightInitialization = new UniformInitialization(0.0,1.0);
    
    
    protected int[] neuronsInHiddenLayers;
    
    
    protected int[] indexesWeightPerLayer;
    
    
    public enum NeuralNetMode { BUILD, TRAINING, RUN };
    
    protected NeuralNetMode neuralNetMode = NeuralNetMode.BUILD;
    
    protected NeuralNet(){
    	
    }
    
    /**
     * init
     * This function initializes the neural network by initializing all of 
     * the underlying layers and their respective neurons based on values set in Spring.
     * 
     * @param numberofinputs Number of Inputs of this Neural Network
     * @param numberofoutputs Number of Outputs of this Neural Network
     * @param numberofhiddenneurons Array containing the number of Neurons in 
     * each of the Hidden Layers
     * @param hiddenAcFnc Array containing the activation function of each 
     * Hidden Layer
     * @param outputAcFnc Activation Function of the Output Layer
     * @param _weightInitialization 
     */
    public void init(){
    	if(hiddenLayers != null) { 
    		numberOfHiddenLayers=hiddenLayers.size();
    	}
    	numberOfOutputs=outputLayer.getNumberOfNeuronsInLayer();
        neuronsInHiddenLayers = new int[numberOfHiddenLayers+1];
        indexesWeightPerLayer = new int[numberOfHiddenLayers+2];  
        for(int i=0;i<=numberOfHiddenLayers;i++){
            if(i==numberOfHiddenLayers){
                neuronsInHiddenLayers[i]=numberOfOutputs;
            }
            else{
                neuronsInHiddenLayers[i]=hiddenLayers.get(i).getNumberOfNeuronsInLayer();
            }
            if(i==0){
                indexesWeightPerLayer[i]=0;
            }
            else{
                indexesWeightPerLayer[i]=indexesWeightPerLayer[i-1]
                        + (neuronsInHiddenLayers[i-1]*
                            ((i==1?numberOfInputs:neuronsInHiddenLayers[i-2])
                            +1));
            }
        }
        if(numberOfHiddenLayers>0){
            indexesWeightPerLayer[numberOfHiddenLayers+1]=
                indexesWeightPerLayer[numberOfHiddenLayers]
                + neuronsInHiddenLayers[numberOfHiddenLayers]
                    *(neuronsInHiddenLayers[numberOfHiddenLayers-1]+1);
        }
        else{
            indexesWeightPerLayer[numberOfHiddenLayers+1]=
                indexesWeightPerLayer[numberOfHiddenLayers] 
                    + neuronsInHiddenLayers[numberOfHiddenLayers]
                        *(numberOfInputs+1);
        }
        input=new ArrayList<>(numberOfInputs);
        inputLayer=new InputLayer(this,numberOfInputs);
        for(int i=0;i<numberOfHiddenLayers;i++){
            if(i==0){
                hiddenLayers.get(i).init(inputLayer.getNumberOfNeuronsInLayer());
                inputLayer.setNextLayer(hiddenLayers.get(i));
            }
            else{
                hiddenLayers.get(i).init(hiddenLayers.get(i-1).getNumberOfNeuronsInLayer());
                hiddenLayers.get(i-1).setNextLayer(hiddenLayers.get(i));
            }
        }
        if(numberOfHiddenLayers>0){
            outputLayer.init(hiddenLayers.get(numberOfHiddenLayers-1).getNumberOfNeuronsInLayer() );
            hiddenLayers.get(numberOfHiddenLayers-1).setNextLayer(outputLayer);
        }
        else{
            outputLayer.init(numberOfInputs);
            inputLayer.setNextLayer(outputLayer);
        }
        setNeuralNetMode(NeuralNetMode.RUN);
    }
    
    /**
     * Feeds an array of real values to the neural network's inputs
     * @param inputs Array of real values to be fed into the neural inputs
     * @throws NeuralException 
     */
    public void setInputs(ArrayList<Double> inputs) throws NeuralException{
        if(inputs.size()==numberOfInputs){
            this.input=inputs;
        } else {
        	throw new NeuralException("Number of inputs is different from the value of the numberOfInputs variable.");
        }
    }
    
    /**
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
     * Gets inputs values of neural net as ArrayList
     * @return array list inputs  
     */
    public ArrayList<Double> getArrayInputs(){
        return input;
    }
    
    /**
     * Gets input value of neuron i
     * @return double input value  
     */
    public Double getInput(int i){
        return input.get(i);
    }
    
    /**
     * Gets inputs values of neural net as double array
     * @return double array of inputs  
     */
    public double[] getInputs(){
        double[] result=new double[numberOfInputs];
        for(int i=0;i<numberOfInputs;i++){
            result[i]=input.get(i);
        }
        return result;
    }
    
    /**
     * This method calculates the output of each layer and forwards all values 
     * to the next layer
     */
    public void calc(){
        inputLayer.setInputs(input);
        inputLayer.calc();
        if(numberOfHiddenLayers>0){
            for(int i=0;i<numberOfHiddenLayers;i++){
                HiddenLayer hl = hiddenLayers.get(i);
                hl.setInputs(hl.getPreviousLayer().getOutputs());
                hl.calc();
            }
        }
        outputLayer.setInputs(outputLayer.getPreviousLayer().getOutputs());
        outputLayer.calc();
        this.output=outputLayer.getOutputs();
    }
    
    /**
     * Returns the neural outputs in the form of ArrayList
     * 
     * @return outputs as array list
     */
    public ArrayList<Double> getArrayOutputs(){
        return output;
    }
    
    /**
     * Returns the neural outputs in the form of array
     * 
     * @return outputs as array
     */
    public double[] getOutputs(){
        double[] _outputs = new double[numberOfOutputs];
        for(int i=0;i<numberOfOutputs;i++){
            _outputs[i]=output.get(i);
        }
        return _outputs;
    }
    
    /**
     * Gets output value of neuron i
     * @return double output value  
     */
    public double getOutput(int i){
        return output.get(i);
    }
    
    /**
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
    
    /**
     * Sets neural net dataset
     * @param neural net dataset object 
     */
    public void setNeuralDataSet(NeuralDataSet _neuralDataSet){
        _neuralDataSet.neuralNet=this;
    }
    
    /**
     * Gets number of hidden layers
     * @return number of hidden layers 
     */
    public int getNumberOfHiddenLayers(){
        return numberOfHiddenLayers;
    }
    
    /**
     * Gets number of inputs
     * @return number of inputs  
     */
    public int getNumberOfInputs(){
        return numberOfInputs;
    }
    
    /**
     * Gets input layer
     * @return input layer object 
     */
    public InputLayer getInputLayer(){
        return inputLayer;
    }
    
    /**
     * Gets hidden layer
     * @return hidden layer object 
     */
    public HiddenLayer getHiddenLayer(int i){
        return hiddenLayers.get(i);
    }
    
    /**
     * Gets hidden layers
     * @return hidden layers as ArrayList 
     */
    public ArrayList<HiddenLayer> getHiddenLayers(){
        return hiddenLayers;
    }
    
    /**
     * Gets output layer
     * @return output layer object 
     */
    public OutputLayer getOutputLayer(){
        return outputLayer;
    }
    
    /**
     * Deactivates bias
     */
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
    
    /**
     * Activates bias
     */
    public void activateBias(){
        for(HiddenLayer hl:hiddenLayers){
            for(Neuron n:hl.getListOfNeurons()){
                n.activateBias();
            }
        }
        for(Neuron n:outputLayer.getListOfNeurons()){
            n.activateBias();
        }
    }
    
    /**
     * Returns if bias is activated or not
     * @return true if it's activated; else, false
     */
    public boolean isBiasActive(){
        return activeBias;
    }
    
    /**
     * Gets weight initialization
     * @return weight init object 
     */
    public WeightInitialization getWeightInitialization(){
        return weightInitialization;
    }
    
    /**
     * Gets weight values
     * @return all weights array 
     */
    public double[] getAllWeights(){
        int numberOfWeights=indexesWeightPerLayer[numberOfHiddenLayers+1];
        double[] weights=new double[numberOfWeights];
        for(int l=0;l<=numberOfHiddenLayers;l++){
            int j=0;
            NeuralLayer nl;
            if(l==numberOfHiddenLayers) // outputlayer
                nl = outputLayer;
            else
                nl = hiddenLayers.get(l);
            
            for(Neuron n:nl.getListOfNeurons()){
                for(int i=0;i<=n.getNumberOfInputs();i++){
                    weights[indexesWeightPerLayer[l]
                           +j*(neuronsInHiddenLayers[l]+1)
                           +i]=n.getWeight(i);
                }
                j++;
            }
        }
        return weights;
    }
    
    /**
     * Gets weight value
     * @param layer index
     * @param neuron index
     * @param input index
     * @return weight 
     */
    public double getWeight(int layer,int neuron,int input){
        if(layer==numberOfHiddenLayers){
            return outputLayer.getWeight(input, neuron);
        }
        else{
            return hiddenLayers.get(layer).getWeight(input, neuron);
        }
    }
    
    /**
     * Gets the total number of weights
     * @return total number of weights 
     */
    public int getTotalNumberOfWeights(){
        int result=0;
        for(HiddenLayer hl:this.hiddenLayers){
            result+=hl.numberOfNeuronsInLayer*(hl.numberOfInputs+1);
        }
        result+=outputLayer.numberOfNeuronsInLayer
                *(outputLayer.numberOfInputs+1);
        return result;
    }
    
    /**
     * Sets neural net mode
     * @param neural net mode 
     */
    public void setNeuralNetMode(NeuralNet.NeuralNetMode _neuralNetMode){
        this.neuralNetMode=_neuralNetMode;
    }
    
    /**
     * Gets neural net mode
     * @return neural net mode 
     */
    public NeuralNetMode getNeuralNetMode(){
        return this.neuralNetMode;
    }
    
    public void setNumberOfInputs(int numberOfInputs) {
		this.numberOfInputs = numberOfInputs;
	}
    
    public void setWeightInitialization(WeightInitialization weightInitialization) {
		this.weightInitialization = weightInitialization;
	}
    public void setHiddenLayers(ArrayList<HiddenLayer> hiddenLayers) {
		this.hiddenLayers = hiddenLayers;
	}
    
    public void setOutputLayer(OutputLayer outputLayer) {
		this.outputLayer = outputLayer;
	}
    
}
