package recommendation;

import java.io.File;
import java.util.List;

import org.apache.mahout.cf.taste.impl.common.LongPrimitiveIterator;
import org.apache.mahout.cf.taste.impl.model.GenericBooleanPrefDataModel;
import org.apache.mahout.cf.taste.impl.model.file.FileDataModel;
import org.apache.mahout.cf.taste.impl.recommender.GenericItemBasedRecommender;
import org.apache.mahout.cf.taste.impl.similarity.PearsonCorrelationSimilarity;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.recommender.RecommendedItem;
import org.apache.mahout.cf.taste.recommender.Recommender;
import org.apache.mahout.cf.taste.similarity.ItemSimilarity;

/*
 * 
 * 基于物品相似度的推荐引擎

 * 
 * 
 */
public class MyItemBasedRecommender {

	public static void main(String[] s) throws Exception {
		DataModel model = new FileDataModel(new File("E:\\test2.txt"));// 构造数据模型，File-based
		// DataModel model=new
		// GenericBooleanPrefDataModel(GenericBooleanPrefDataModel.toDataMap(model1));
		ItemSimilarity similarity = new PearsonCorrelationSimilarity(model);// 计算内容相似度
		Recommender recommender = new GenericItemBasedRecommender(model,
				similarity);// 构造推荐引擎

		LongPrimitiveIterator iter = model.getUserIDs();

		while (iter.hasNext()) {
			long uid = iter.nextLong();
			List<RecommendedItem> list = recommender.recommend(uid, 3);
			System.out.printf("uid:%s", uid);
			for (RecommendedItem ritem : list) {
				System.out.printf("(%s,%f)", ritem.getItemID(),
						ritem.getValue());
			}
			System.out.println();
		}

	}

}