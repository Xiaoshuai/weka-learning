package com.weka.learning;

import java.io.FileReader;
import java.io.IOException;

import weka.core.Instances;

public class L01Instances {

	public static void main(String[] args) throws IOException {
		FileReader fr = new FileReader(L01Instances.class.getClassLoader()
				.getResource("com/weka/learning/contact-lenses.arff").getFile());
		Instances instances = new Instances(fr);

		// Out put all the arff data
		System.out.println(instances);
		System.out.println(">----------------------------------------<");

		for (int i = 0; i < instances.numInstances(); i++) {
			System.out.println("[instance " + i + "]" + instances.instance(i));
		}
	}

}
