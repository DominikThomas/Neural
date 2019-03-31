package com.dominikthomas.neuralnet.beans;

import com.dominikthomas.neuralnet.math.IActivationFunction;

/**
 *
 * This class extends from NeuralLayer and represents a hidden layer in a Neural 
 * Network
 * 
 * @authors Alan de Souza, Fabio Soares 
 * @version 0.1
 * 
 */
public class HiddenLayer extends NeuralLayer {
    
    /**
     * HiddenLayer init
     * 
     * @param numberofinputs Number of inputs in this layer
     * @see HiddenLayer
     */
    public void init(int numberofinputs){
    	numberOfInputs=numberofinputs;
		super.init();
    }    
    public HiddenLayer() {
    }    
    
    /**
     * This method links this layer to a previous layer in the Neural Network
     * 
     * @param previous Previous Neural Layer
     * @see HiddenLayer
     */
    @Override
    public void setPreviousLayer(NeuralLayer previous){
        this.previousLayer=previous;
        if(previous.nextLayer!=this)
            previous.setNextLayer(this);
    }
    
    /**
     * This method links this layer to a next layer in the Neural Network
     * 
     * @param next Next Neural Layer
     * @see HiddenLayer
     */
    @Override
    public void setNextLayer(NeuralLayer next){
        nextLayer=next;
        if(next.previousLayer!=this)
            next.setPreviousLayer(this);
    }
    
}
