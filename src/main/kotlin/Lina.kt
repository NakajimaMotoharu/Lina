import org.nd4j.linalg.api.ops.DynamicCustomOp
import org.nd4j.linalg.api.ops.impl.transforms.custom.Svd
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.indexing.NDArrayIndex

class Lina {
	companion object{
		// ベクトルaとbに対し、和c = a + bを計算
		fun addVecVec(a: FloatArray, b: FloatArray): FloatArray {
			val va = Nd4j.create(a)
			val vb = Nd4j.create(b)
			return va.add(vb).toFloatVector()
		}

		// ベクトルaとbに対し、差c = a - bを計算
		fun subVecVec(a: FloatArray, b: FloatArray): FloatArray {
			val va = Nd4j.create(a)
			val vb = Nd4j.create(b)
			return va.sub(vb).toFloatVector()
		}

		// ベクトルaとbの内積をFloat型で返す
		fun ipVecVec(a: FloatArray, b: FloatArray): Float {
			val va = Nd4j.create(a)
			val vb = Nd4j.create(b)
			return Nd4j.linalg.matmul(va, vb).getFloat(0)
		}

		// 行列AとBに対し、積C = A + Bを計算
		fun addMatMat(a: Array<FloatArray>, b: Array<FloatArray>): Array<FloatArray> {
			val ma = Nd4j.create(a)
			val mb = Nd4j.create(b)
			return ma.add(mb).toFloatMatrix()
		}

		// 行列AとBに対し、差C = A - Bを計算
		fun subMatMat(a: Array<FloatArray>, b: Array<FloatArray>): Array<FloatArray> {
			val ma = Nd4j.create(a)
			val mb = Nd4j.create(b)
			return ma.sub(mb).toFloatMatrix()
		}

		// 行列AとBに対し、積C = A * Bを計算
		fun mulMatMat(a: Array<FloatArray>, b: Array<FloatArray>): Array<FloatArray>{
			val m1 = Nd4j.create(a)
			val m2 = Nd4j.create(b)
			val m3 = m1.mmul(m2)
			return m3.toFloatMatrix()
		}

		// 行列Aとベクトルbに対し、積c = A * bを計算
		fun mulMatVec(a: Array<FloatArray>, b: FloatArray): FloatArray {
			val ma = Nd4j.create(a)
			val mb = Nd4j.create(b)
			return ma.mmul(mb).toFloatVector()
		}

		// 行列Aと実数kに対し、実数倍C = k * Aを計算
		fun mulScaMat(a: Array<FloatArray>, k: Float): Array<FloatArray>{
			return Nd4j.create(a).mul(k).toFloatMatrix()
		}

		// ベクトルaと実数kに対し、実数倍c = k * aを計算
		fun mulScaVec(a: FloatArray, b: Float): FloatArray{
			return Nd4j.create(a).mul(b).toFloatVector()
		}

		// 行列Aに対し、Aの転置C = A^tを計算
		fun tMat(a: Array<FloatArray>): Array<FloatArray>{
			return Nd4j.create(a).transpose().toFloatMatrix()
		}

		// 行列Aに対し、Aの逆行列C = A^{-1}を計算
		fun invMat(a: Array<FloatArray>): Array<FloatArray>{
			return Nd4j.linalg.matrixInverse(Nd4j.create(a)).toFloatMatrix()
		}

		// 正方行列Aと定数ベクトルbに対し、連立方程式A * x = bの解xを求める
		fun solveSeq(a: Array<FloatArray>, b: FloatArray): FloatArray{
			val ma = Nd4j.create(a)
			val mb = Nd4j.create(vectorToMatrix(b))
			return Nd4j.linalg.solve(ma, mb).toFloatVector()
		}

		// 対称行列Aの固有値/固有ベクトルを求め、固有値を大きい方から小さい方にソートしてベクトルpair.firstに入れ、
		// 対応する固有ベクトル(大きさ1に規格化済)を列方向に並べて行列pair.secondに入れる
		fun symEigen(a: Array<FloatArray>): Pair<FloatArray, Array<FloatArray>>{
			val eig = Nd4j.linalg.eig(Nd4j.create(a))

			val lambda = eig[0].get(NDArrayIndex.all(), NDArrayIndex.point(0)).toFloatVector()
			val vector = eig[1].get(NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.point(0)).toFloatMatrix()

			return eigSort(lambda, vector)
		}

		// 正定値行列Aのコレスキー分解A = U^t * Uを求める
		// Uは上三角行列として返される
		fun chDecomp(a: Array<FloatArray>): Array<FloatArray>{
			val ma = Nd4j.create(a)
			return Nd4j.linalg.cholesky(ma).toFloatMatrix()
		}

		// 行列Aに対し、Aの行列式C = det(A)を計算
		fun detMat(a: Array<FloatArray>): Float{
			return Nd4j.linalg.matrixDeterminant(Nd4j.create(a)).getFloat(0)
		}

		// ベクトルaに対し、aのユークリッド距離を計算
		fun normVec(a: FloatArray): Float {
			return Nd4j.norm2(Nd4j.create(a)).getFloat(0)
		}

		// 行列Aに対し、特異値分解N = U * SIGMA * V^tを計算し、
		// U, SIGMA, V^tが返される
		fun svdMat(a: Array<FloatArray>): Pair<FloatArray, Pair<Array<FloatArray>, Array<FloatArray>>>{
			val ma = Nd4j.create(a)
			val act = DynamicCustomOp.builder("svd").addInputs(ma).addIntegerArguments(1, 1, Svd.DEFAULT_SWITCHNUM).build()
			val svd = Nd4j.getExecutioner().exec(act)

			val u = svd[0].toFloatVector()
			val s = svd[1].toFloatMatrix()
			val v = svd[2].toFloatMatrix()

			return Pair(u, Pair(s, v))
		}

		// 行列Aに対し、svdMatを利用して、固有値/固有ベクトルを求め
		// 固有値を大きい方から小さい方にソートしてベクトルpair.firstに入れ、
		// 対応する固有ベクトル(大きさ1に規格化済)を列方向に並べて行列pair.secondに入れる
		fun svdSymEigen(a: Array<FloatArray>): Pair<FloatArray, Array<FloatArray>>{
			val (lambda, pair) = svdMat(a)
			val (s, vector) = pair

			for (i in 0 until vector[0].size){
				val ip = ipVecVec(matrixToVector(s, i), matrixToVector(vector, i))
				if (ip < 0f){
					lambda[i] *= -1f
					for (j in 0 until vector.size){
						vector[j][i] *= -1f
					}
				}
			}

			return eigSort(lambda, vector)
		}

		// ベクトルを行列形式に
		fun vectorToMatrix(v: FloatArray): Array<FloatArray>{
			val ans: Array<FloatArray> = Array(v.size) { FloatArray(1) }

			for (i in 0 until v.size){
				ans[i][0] = v[i]
			}

			return ans
		}

		// 行列をベクトル形式に
		fun matrixToVector(m: Array<FloatArray>, n: Int): FloatArray{
			val ans = FloatArray(m.size)

			for (i in 0 until m.size){
				ans[i] = m[i][n]
			}

			return ans
		}

		// ベクトル表示(文字列付き)
		fun printVec(str: String, vec: FloatArray) {
			printMat(str, vectorToMatrix(vec))
		}

		// 行列表示(文字列付き)
		fun printMat(str: String, matrix: Array<FloatArray>) {
			var strSize: Int = 0
			for (i in 0 until matrix.size) {
				for (j in 0 until matrix[0].size) {
					val str: String = String.format("%f", matrix[i][j])
					if (strSize < str.length) {
						strSize = str.length
					}
				}
			}

			val fmt: String = "%" + strSize + "f"
			var spc = ""
			for (i in  0 until str.length){
				spc += " "
			}

			for (i in 0 until matrix.size) {
				if (i == 0) print(str) else print(spc)
				print("|")
				for (j in 0 until matrix[0].size) {
					print(String.format(fmt, matrix[i][j]))
					if (j != matrix[0].size - 1) {
						print(" ")
					} else {
						println("|")
					}
				}
			}
		}

		// ベクトル表示(文字列なし)
		fun printVec(vec: FloatArray){
			printVec("", vec)
		}

		// 行列表示(文字列なし)
		fun printMat(matrix: Array<FloatArray>){
			printMat("", matrix)
		}

		// 固有値と固有ベクトルを大きい方から小さい方にソート
		private fun eigSort(lambda: FloatArray, vector: Array<FloatArray>): Pair<FloatArray, Array<FloatArray>> {
			val sortedIndices = lambda.indices.sortedByDescending { lambda[it] }
			val sortedLambda = sortedIndices.map { lambda[it] }.toFloatArray()
			val sortedVector = Array(vector.size) { i ->
				FloatArray(vector[0].size) { j ->
					vector[i][sortedIndices[j]]
				}
			}

			return Pair(sortedLambda, sortedVector)
		}
	}
}