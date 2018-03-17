package com.dominikthomas.neuralnet;

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
 * @version 0.2
 */
public class NeuralNetSpring {
	@SuppressWarnings("resource")
	public static void main(String[] args) {
 
		// loading the definitions from the given XML file
		ApplicationContext context = new ClassPathXmlApplicationContext(
				"neuralnet/applicationContext.xml");

		RandomNumberGenerator.setSeed(0);
 
		NeuralNet nn = (NeuralNet) context
				.getBean("neuralNet");
        
		try {
			nn.init();
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
        
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
