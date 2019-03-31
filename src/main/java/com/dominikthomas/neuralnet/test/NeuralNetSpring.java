package com.dominikthomas.neuralnet.test;

import java.util.Arrays;

import org.springframework.context.ApplicationContext;
import org.springframework.context.support.ClassPathXmlApplicationContext;

import com.dominikthomas.neuralnet.math.RandomNumberGenerator;
import com.dominikthomas.neuralnet.beans.NeuralNet;

/**
 *
 * NeuralNetConsoleTest
 * This class is solely used for creating and testing your very first Neural 
 * Network in Java 
 *
 * @author Alan de Souza, Fábio Soares, Dominik Thomas
 */
public class NeuralNetSpring {
	@SuppressWarnings("resource")
	public static void main(String[] args) {
 
		RandomNumberGenerator.setSeed(2);
		
		// loading the definitions from the given XML file
		ApplicationContext context = new ClassPathXmlApplicationContext(
				"neuralnetspring/applicationContext.xml");
 
		NeuralNet nn = (NeuralNet) context
				.getBean("neuralNet");
		
        double [] neuralInput = { 1.5 , 0.5 };
        
        double [] neuralOutput;
        nn.setInputs(neuralInput);
        nn.calc();
        neuralOutput=nn.getOutputs();
        System.out.println(Arrays.toString(neuralOutput));

        neuralInput[0] = 1.0;
        neuralInput[1] = 2.1;
        
        nn.setInputs(neuralInput);
        nn.calc();
        neuralOutput=nn.getOutputs();
        System.out.println(Arrays.toString(neuralOutput));
        
    }
}
