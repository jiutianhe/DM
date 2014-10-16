package recommendation;

import java.io.File;
import java.util.List;

import org.apache.mahout.cf.taste.impl.common.LongPrimitiveIterator;
import org.apache.mahout.cf.taste.impl.model.file.FileDataModel;
import org.apache.mahout.cf.taste.impl.recommender.CachingRecommender;
import org.apache.mahout.cf.taste.impl.recommender.slopeone.SlopeOneRecommender;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.recommender.RecommendedItem;
import org.apache.mahout.cf.taste.recommender.Recommender;

/*
 * 基于Slop One的推荐引擎

 * 
 * 基于用户和基于内容是最常用最容易理解的两种推荐策略，但在大数据量时，它们的计算量会很大，从而导致推荐效率较差。因此 Mahout 还提供了一种更加轻量级的 CF 推荐策略：Slope One。
 Slope One 是有 Daniel Lemire 和 Anna Maclachlan 在 2005 年提出的一种对基于评分的协同过滤推荐引擎的改进方法，下面简单介绍一下它的基本思想。
 假设系统对于物品 A，物品 B 和物品 C 的平均评分分别是 3，4 和 4。基于 Slope One 的方法会得到以下规律：
 •用户对物品 B 的评分 = 用户对物品 A 的评分 + 1
 •用户对物品 B 的评分 = 用户对物品 C 的评分
 基于以上的规律，我们可以对用户 A 和用户 B 的打分进行预测：
 •对用户 A，他给物品 A 打分 4，那么我们可以推测他对物品 B 的评分是 5，对物品 C 的打分也是 5。
 •对用户 B，他给物品 A 打分 2，给物品 C 打分 4，根据第一条规律，我们可以推断他对物品 B 的评分是 3；而根据第二条规律，推断出评分是 4。当出现冲突时，我们可以对各种规则得到的推断进行就平均，所以给出的推断是 3.5。
 这就是 Slope One 推荐的基本原理，它将用户的评分之间的关系看作简单的线性关系：
 Y = mX + b;
 当 m = 1 时就是 Slope One，也就是我们刚刚展示的例子。
 * 
 * 
 * 
 */

public class MySlopeOneRecommender {

	public static void main(String[] s) throws Exception {

		DataModel model = new FileDataModel(new File("E:\\test2.txt"));// 构造数据模型
		Recommender recommender = new CachingRecommender(
				new SlopeOneRecommender(model));// 构造推荐引擎

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