package com.netease.ebot.weka;

import weka.classifiers.Classifier;
import weka.classifiers.trees.J48;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.StringToWordVector;

import java.io.File;
import java.io.FileReader;
import java.io.Serializable;

/**
 * Java program for classifying short text messages into two classes 'miss' and
 * 'hit'.
 * <p/>
 * See also wiki article <a
 * href="http://weka.wiki.sourceforge.net/MessageClassifier"
 * >MessageClassifier</a>.
 */
public class MessageClassifier implements Serializable {

	/** for serialization. */
	private static final long serialVersionUID = -123455813150452885L;

	/** The training data gathered so far. */
	private Instances m_Data = null;

	/** The filter used to generate the word counts. */
	private StringToWordVector m_Filter = new StringToWordVector();

	/** The actual classifier. */
	private Classifier m_Classifier = new J48();

	/** Whether the model is up to date. */
	private boolean m_UpToDate;

	/**
	 * Constructs empty training dataset.
	 */
	public MessageClassifier() {
		String nameOfDataset = "MessageClassificationProblem";

		// Create vector of attributes.
		FastVector attributes = new FastVector(2);

		// Add attribute for holding messages.
		attributes.addElement(new Attribute("Message", (FastVector) null));

		// Add class attribute.
		FastVector classValues = new FastVector(2);
		classValues.addElement("miss");
		classValues.addElement("hit");
		attributes.addElement(new Attribute("Class", classValues));

		// Create dataset with initial capacity of 100, and set index of class.
		m_Data = new Instances(nameOfDataset, attributes, 100);
		m_Data.setClassIndex(m_Data.numAttributes() - 1);
	}

	/**
	 * Updates model using the given training message.
	 * 
	 * @param message
	 *            the message content
	 * @param classValue
	 *            the class label
	 */
	public void updateData(String message, String classValue) {
		// Make message into instance.
		Instance instance = makeInstance(message, m_Data);

		// Set class value for instance.
		instance.setClassValue(classValue);

		// Add instance to training data.
		m_Data.add(instance);

		m_UpToDate = false;
	}

	/**
	 * Classifies a given message.
	 * 
	 * @param message
	 *            the message content
	 * @throws Exception
	 *             if classification fails
	 */
	public void classifyMessage(String message) throws Exception {
		// Check whether classifier has been built.
		if (m_Data.numInstances() == 0)
			throw new Exception("No classifier available.");

		// Check whether classifier and filter are up to date.
		if (!m_UpToDate) {
			// Initialize filter and tell it about the input format.
			m_Filter.setInputFormat(m_Data);

			// Generate word counts from the training data.
			Instances filteredData = Filter.useFilter(m_Data, m_Filter);

			// Rebuild classifier.
			m_Classifier.buildClassifier(filteredData);

			m_UpToDate = true;
		}

		// Make separate little test set so that message
		// does not get added to string attribute in m_Data.
		Instances testset = m_Data.stringFreeStructure();

		// Make message into test instance.
		Instance instance = makeInstance(message, testset);

		// Filter instance.
		m_Filter.input(instance);
		Instance filteredInstance = m_Filter.output();

		// Get index of predicted class value.
		double predicted = m_Classifier.classifyInstance(filteredInstance);

		// Output class value.
		System.err.println("Message classified as : " + m_Data.classAttribute().value((int) predicted));
	}

	/**
	 * Method that converts a text message into an instance.
	 * 
	 * @param text
	 *            the message content to convert
	 * @param data
	 *            the header information
	 * @return the generated Instance
	 */
	private Instance makeInstance(String text, Instances data) {
		// Create instance of length two.
		Instance instance = new DenseInstance(2);

		// Set value for message attribute
		Attribute messageAtt = data.attribute("Message");
		instance.setValue(messageAtt, messageAtt.addStringValue(text));

		// Give instance access to attribute information from the dataset.
		instance.setDataset(data);

		return instance;
	}

	public static void main(String[] args) throws Exception {
		String modelName = MessageClassifier.class.getResource("").getFile() +  "messageclassifier.model";
		File modelFile = new File(modelName);
		if (modelFile.exists()) {
			modelFile.delete();
		}
		
		// 第一次为训练
		String messageName = "lenses-miss.txt";
		String classValue = "miss";
		FileReader m = new FileReader(MessageClassifier.class.getResource(messageName).getFile());
		StringBuffer message = new StringBuffer();
		int l;
		while ((l = m.read()) != -1)
			message.append((char) l);
		m.close();
		// 创建模型文件
		MessageClassifier messageCl = new MessageClassifier();
		// 处理数据至模型
		messageCl.updateData(message.toString(), classValue);
		// 更新模型
		SerializationHelper.write(modelName, messageCl);
		m.close();

		// 第二次为训练
		messageName = "lenses-hit.txt";
		classValue = "hit";
		m = new FileReader(MessageClassifier.class.getResource(messageName).getFile());
		message = new StringBuffer();
		while ((l = m.read()) != -1)
			message.append((char) l);
		m.close();
		// 读取模型
		messageCl = (MessageClassifier) SerializationHelper.read(modelName);
		// 处理数据至模型
		messageCl.updateData(message.toString(), classValue);
		// 更新模型
		SerializationHelper.write(modelName, messageCl);
		m.close();

		// 第三次为测试
		messageName = "lenses-test.txt";
		classValue = "";
		m = new FileReader(MessageClassifier.class.getResource(messageName).getFile());
		message = new StringBuffer();
		while ((l = m.read()) != -1)
			message.append((char) l);
		m.close();
		// 读取模型
		messageCl = (MessageClassifier) SerializationHelper.read(modelName);
		// 使用模型分类
		messageCl.classifyMessage(message.toString());
		m.close();
	}
}
