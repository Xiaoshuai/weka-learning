package com.weka.learning;

import java.io.FileReader;
import java.util.Random;

import weka.attributeSelection.CfsSubsetEval;
import weka.attributeSelection.GreedyStepwise;
import weka.classifiers.Evaluation;
import weka.classifiers.meta.AttributeSelectedClassifier;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.supervised.attribute.AttributeSelection;

public class L04AttributeSelection {

	public static void main(String[] args) throws Exception {

		FileReader fr = new FileReader(L01Instances.class.getClassLoader()
				.getResource("com/weka/learning/soybean.arff").getFile());
		Instances instances = new Instances(fr);
		instances.setClassIndex(instances.numAttributes() - 1);
		System.out.println(">--------------------selectAttUseFilter--------------------<");
		selectAttUseFilter(instances);

		System.out.println(">--------------------selectAttUseMC--------------------<");
		selectAttUseMC(instances);
	}

	public static void selectAttUseFilter(Instances instances) throws Exception {
		AttributeSelection filter = new AttributeSelection(); // package weka.filters.supervised.attribute!
		CfsSubsetEval eval = new CfsSubsetEval();
		GreedyStepwise search = new GreedyStepwise();
		filter.setEvaluator(eval);
		filter.setSearch(search);
		filter.setInputFormat(instances);

		System.out.println("number of instance attribute = " + instances.numAttributes());

		Instances selectedIns = Filter.useFilter(instances, filter);
		System.out.println("number of selected instance attribute = " + selectedIns.numAttributes());
	}

	public static void selectAttUseMC(Instances instances) throws Exception {
		AttributeSelectedClassifier classifier = new AttributeSelectedClassifier();
		CfsSubsetEval eval = new CfsSubsetEval();
		GreedyStepwise search = new GreedyStepwise();
		J48 base = new J48();
		classifier.setClassifier(base);
		classifier.setEvaluator(eval);
		classifier.setSearch(search);
		// 10-fold cross-validation
		Evaluation evaluation = new Evaluation(instances);
		evaluation.crossValidateModel(classifier, instances, 10, new Random(1));
		System.out.println(evaluation.toSummaryString());
	}

}
