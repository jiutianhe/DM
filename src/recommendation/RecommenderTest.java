package recommendation;

import java.io.File;
import java.io.IOException;
import java.util.List;

import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.eval.IRStatistics;
import org.apache.mahout.cf.taste.eval.RecommenderBuilder;
import org.apache.mahout.cf.taste.eval.RecommenderEvaluator;
import org.apache.mahout.cf.taste.eval.RecommenderIRStatsEvaluator;
import org.apache.mahout.cf.taste.impl.common.LongPrimitiveIterator;
import org.apache.mahout.cf.taste.impl.eval.AverageAbsoluteDifferenceRecommenderEvaluator;
import org.apache.mahout.cf.taste.impl.eval.GenericRecommenderIRStatsEvaluator;
import org.apache.mahout.cf.taste.impl.model.file.FileDataModel;
import org.apache.mahout.cf.taste.impl.neighborhood.NearestNUserNeighborhood;
import org.apache.mahout.cf.taste.impl.recommender.GenericItemBasedRecommender;
import org.apache.mahout.cf.taste.impl.recommender.GenericUserBasedRecommender;
import org.apache.mahout.cf.taste.impl.recommender.svd.ALSWRFactorizer;
import org.apache.mahout.cf.taste.impl.recommender.svd.Factorizer;
import org.apache.mahout.cf.taste.impl.recommender.svd.SVDRecommender;
import org.apache.mahout.cf.taste.impl.similarity.EuclideanDistanceSimilarity;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.neighborhood.UserNeighborhood;
import org.apache.mahout.cf.taste.recommender.RecommendedItem;
import org.apache.mahout.cf.taste.recommender.Recommender;
import org.apache.mahout.cf.taste.similarity.ItemSimilarity;
import org.apache.mahout.cf.taste.similarity.UserSimilarity;
import org.apache.mahout.common.RandomUtils;

/**
 * 根据文章中关于基于用户、物品、SVD的实现代码;基于mahout0.9
 * 
 * @author J
 *
 */
public class RecommenderTest {

	private final static int NEIGHBORHOOD_NUM = 2;
	private final static int RECOMMENDER_NUM = 3;
	private final static String USER_CF = "userCF";
	private final static String ITEM_CF = "itemCF";
	private final static String SVD = "svd";

	public static void main(String[] args) throws IOException, TasteException {

		RandomUtils.useTestSeed();

		String file = "datafile/item.csv";

		DataModel model = new FileDataModel(new File(file));

		collaborativeFiltering(model, USER_CF);
		collaborativeFiltering(model, ITEM_CF);
		collaborativeFiltering(model, SVD);

	}

	public static void collaborativeFiltering(DataModel model, String algorithm)
			throws TasteException {
		// 评估器
		RecommenderEvaluator evaluator = new AverageAbsoluteDifferenceRecommenderEvaluator();
		// 召回率和查准率 评估器
		RecommenderIRStatsEvaluator statEvaluator = new GenericRecommenderIRStatsEvaluator();
		// 协同过滤器
		RecommenderBuilder builder = getRecommenderBuilder(algorithm);

		double score = evaluator.evaluate(builder, null, model, 0.7, 1.0);

		IRStatistics stats = statEvaluator.evaluate(builder, null, model, null,
				2, GenericRecommenderIRStatsEvaluator.CHOOSE_THRESHOLD, 1.0);

		LongPrimitiveIterator it = model.getUserIDs();

		System.out.println("socre:" + score);
		System.out.println("precision:" + stats.getPrecision() + "  recall:"
				+ stats.getRecall());

		while (it.hasNext()) {
			long uid = it.next();
			// 得到要推荐的项集合
			List<RecommendedItem> recommendations = builder.buildRecommender(
					model).recommend(uid, RECOMMENDER_NUM);
			System.out.print("uid:" + uid + ",");
			for (RecommendedItem recommendation : recommendations) {
				System.out.print("(" + recommendation.getItemID() + ","
						+ recommendation.getValue() + ")");
			}
			System.out.println();
		}
	}

	public static RecommenderBuilder getRecommenderBuilder(String algorithm) {
		// 基于用户的协同过滤算法
		if ("userCF".equals(algorithm)) {
			return new RecommenderBuilder() {

				@Override
				public Recommender buildRecommender(DataModel dataModel)
						throws TasteException {
					UserSimilarity similarity = new EuclideanDistanceSimilarity(
							dataModel);
					UserNeighborhood neighborhood = new NearestNUserNeighborhood(
							NEIGHBORHOOD_NUM, similarity, dataModel);

					return new GenericUserBasedRecommender(dataModel,
							neighborhood, similarity);
				}
			};
			// 基于物品的协同过滤算法
		} else if ("itemCF".equals(algorithm)) {
			return new RecommenderBuilder() {

				@Override
				public Recommender buildRecommender(DataModel dataModel)
						throws TasteException {
					ItemSimilarity similarity = new EuclideanDistanceSimilarity(
							dataModel);
					return new GenericItemBasedRecommender(dataModel,
							similarity);
				}
			};
			// 基于SVD的协同过滤算法
		} else if ("svd".equals(algorithm)) {
			return new RecommenderBuilder() {

				@Override
				public Recommender buildRecommender(DataModel dataModel)
						throws TasteException {
					Factorizer factorizer = new ALSWRFactorizer(dataModel, 10,
							0.05, 10);
					// Factorizer factorizer = new
					// SVDPlusPlusFactorizer(dataModel, 10, 10);
					return new SVDRecommender(dataModel, factorizer);
				}
			};
		}

		return null;
	}
}
