import org.nd4j.linalg.api.ops.DynamicCustomOp
import org.nd4j.linalg.api.ops.impl.transforms.custom.Svd
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.indexing.NDArrayIndex

class Lina {
	companion object {
		// ベクトルaとbに対し、和c = a + bを計算
		/**
		 * Calculates the sum of two vectors a and b.
		 * @param a First vector
		 * @param b Second vector
		 * @return Resultant vector c = a + b
		 */
		fun addVecVec(a: FloatArray, b: FloatArray): FloatArray {
			val va = Nd4j.create(a)
			val vb = Nd4j.create(b)
			return va.add(vb).toFloatVector()
		}

		// ベクトルaとbに対し、差c = a - bを計算
		/**
		 * Calculates the difference of two vectors a and b.
		 * @param a First vector
		 * @param b Second vector
		 * @return Resultant vector c = a - b
		 */
		fun subVecVec(a: FloatArray, b: FloatArray): FloatArray {
			val va = Nd4j.create(a)
			val vb = Nd4j.create(b)
			return va.sub(vb).toFloatVector()
		}

		// ベクトルaとbの内積をFloat型で返す
		/**
		 * Returns the inner product of two vectors a and b.
		 * @param a First vector
		 * @param b Second vector
		 * @return Inner product of a and b
		 */
		fun ipVecVec(a: FloatArray, b: FloatArray): Float {
			val va = Nd4j.create(a)
			val vb = Nd4j.create(b)
			return Nd4j.linalg.matmul(va, vb).getFloat(0)
		}

		// 行列AとBに対し、積C = A + Bを計算
		/**
		 * Calculates the sum of two matrices A and B.
		 * @param a First matrix
		 * @param b Second matrix
		 * @return Resultant matrix C = A + B
		 */
		fun addMatMat(a: Array<FloatArray>, b: Array<FloatArray>): Array<FloatArray> {
			val ma = Nd4j.create(a)
			val mb = Nd4j.create(b)
			return ma.add(mb).toFloatMatrix()
		}

		// 行列AとBに対し、差C = A - Bを計算
		/**
		 * Calculates the difference of two matrices A and B.
		 * @param a First matrix
		 * @param b Second matrix
		 * @return Resultant matrix C = A - B
		 */
		fun subMatMat(a: Array<FloatArray>, b: Array<FloatArray>): Array<FloatArray> {
			val ma = Nd4j.create(a)
			val mb = Nd4j.create(b)
			return ma.sub(mb).toFloatMatrix()
		}

		// 行列AとBに対し、積C = A * Bを計算
		/**
		 * Calculates the product of two matrices A and B.
		 * @param a First matrix
		 * @param b Second matrix
		 * @return Resultant matrix C = A * B
		 */
		fun mulMatMat(a: Array<FloatArray>, b: Array<FloatArray>): Array<FloatArray> {
			val m1 = Nd4j.create(a)
			val m2 = Nd4j.create(b)
			val m3 = m1.mmul(m2)
			return m3.toFloatMatrix()
		}

		// 行列Aとベクトルbに対し、積c = A * bを計算
		/**
		 * Calculates the product of matrix A and vector b.
		 * @param a Matrix A
		 * @param b Vector b
		 * @return Resultant vector c = A * b
		 */
		fun mulMatVec(a: Array<FloatArray>, b: FloatArray): FloatArray {
			val ma = Nd4j.create(a)
			val mb = Nd4j.create(b)
			return ma.mmul(mb).toFloatVector()
		}

		// 行列Aと実数kに対し、実数倍C = k * Aを計算
		/**
		 * Calculates the product of a scalar k and matrix A.
		 * @param a Matrix A
		 * @param k Scalar value
		 * @return Resultant matrix C = k * A
		 */
		fun mulScaMat(a: Array<FloatArray>, k: Float): Array<FloatArray> {
			return Nd4j.create(a).mul(k).toFloatMatrix()
		}

		// ベクトルaと実数kに対し、実数倍c = k * aを計算
		/**
		 * Calculates the product of a scalar k and vector a.
		 * @param a Vector a
		 * @param b Scalar value
		 * @return Resultant vector c = k * a
		 */
		fun mulScaVec(a: FloatArray, b: Float): FloatArray {
			return Nd4j.create(a).mul(b).toFloatVector()
		}

		// 行列Aに対し、Aの転置C = A^tを計算
		/**
		 * Calculates the transpose of matrix A.
		 * @param a Matrix A
		 * @return Transposed matrix C = A^t
		 */
		fun tMat(a: Array<FloatArray>): Array<FloatArray> {
			return Nd4j.create(a).transpose().toFloatMatrix()
		}

		// 行列Aに対し、Aの逆行列C = A^{-1}を計算
		/**
		 * Calculates the inverse of matrix A.
		 * @param a Matrix A
		 * @return Inverse matrix C = A^{-1}
		 */
		fun invMat(a: Array<FloatArray>): Array<FloatArray> {
			return Nd4j.linalg.matrixInverse(Nd4j.create(a)).toFloatMatrix()
		}

		// 正方行列Aと定数ベクトルbに対し、連立方程式A * x = bの解xを求める
		/**
		 * Solves the system of equations A * x = b.
		 * @param a Coefficient matrix A
		 * @param b Constant vector b
		 * @return Solution vector x
		 */
		fun solveSeq(a: Array<FloatArray>, b: FloatArray): FloatArray {
			val ma = Nd4j.create(a)
			val mb = Nd4j.create(vectorToMatrix(b))
			return Nd4j.linalg.solve(ma, mb).toFloatVector()
		}

		// 対称行列Aの固有値/固有ベクトルを求め、固有値を大きい方から小さい方にソートしてベクトルpair.firstに入れ、
		// 対応する固有ベクトル(大きさ1に規格化済)を列方向に並べて行列pair.secondに入れる
		/**
		 * Calculates the eigenvalues and eigenvectors of a symmetric matrix A.
		 * @param a Symmetric matrix A
		 * @return A Pair containing eigenvalues sorted in descending order and corresponding normalized eigenvectors
		 */
		fun symEigen(a: Array<FloatArray>): Pair<FloatArray, Array<FloatArray>> {
			val eig = Nd4j.linalg.eig(Nd4j.create(a))

			val lambda = eig[0].get(NDArrayIndex.all(), NDArrayIndex.point(0)).toFloatVector()
			val vector = eig[1].get(NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.point(0)).toFloatMatrix()

			return eigSort(lambda, vector)
		}

		// 正定値行列Aのコレスキー分解A = U^t * Uを求める
		// Uは上三角行列として返される
		/**
		 * Performs Cholesky decomposition on a positive definite matrix A.
		 * @param a Positive definite matrix A
		 * @return Upper triangular matrix U such that A = U^t * U
		 */
		fun chDecomp(a: Array<FloatArray>): Array<FloatArray> {
			val ma = Nd4j.create(a)
			return Nd4j.linalg.cholesky(ma).toFloatMatrix()
		}

		// 行列Aに対し、Aの行列式C = det(A)を計算
		/**
		 * Calculates the determinant of matrix A.
		 * @param a Matrix A
		 * @return Determinant of A
		 */
		fun detMat(a: Array<FloatArray>): Float {
			return Nd4j.linalg.matrixDeterminant(Nd4j.create(a)).getFloat(0)
		}

		// ベクトルaに対し、aのユークリッド距離を計算
		/**
		 * Calculates the Euclidean norm of vector a.
		 * @param a Vector a
		 * @return Euclidean distance (norm) of vector a
		 */
		fun normVec(a: FloatArray): Float {
			return Nd4j.norm2(Nd4j.create(a)).getFloat(0)
		}

		// 行列Aに対し、特異値分解N = U * SIGMA * V^tを計算し、
		// U, SIGMA, V^tが返される
		/**
		 * Performs Singular Value Decomposition (SVD) on matrix A.
		 * @param a Matrix A
		 * @return A Pair containing U, SIGMA, and V^t from the SVD
		 */
		fun svdMat(a: Array<FloatArray>): Pair<FloatArray, Pair<Array<FloatArray>, Array<FloatArray>>> {
			val ma = Nd4j.create(a)
			val act =
				DynamicCustomOp.builder("svd").addInputs(ma).addIntegerArguments(1, 1, Svd.DEFAULT_SWITCHNUM).build()
			val svd = Nd4j.getExecutioner().exec(act)

			val u = svd[0].toFloatVector()
			val s = svd[1].toFloatMatrix()
			val v = svd[2].toFloatMatrix()

			return Pair(u, Pair(s, v))
		}

		// 行列Aに対し、svdMatを利用して、固有値/固有ベクトルを求め
		// 固有値を大きい方から小さい方にソートしてベクトルpair.firstに入れ、
		// 対応する固有ベクトル(大きさ1に規格化済)を列方向に並べて行列pair.secondに入れる
		/**
		 * Calculates eigenvalues and eigenvectors using SVD and sorts them.
		 * @param a Symmetric matrix A
		 * @return A Pair containing sorted eigenvalues and corresponding normalized eigenvectors
		 */
		fun svdSymEigen(a: Array<FloatArray>): Pair<FloatArray, Array<FloatArray>> {
			val (lambda, pair) = svdMat(a)
			val (s, vector) = pair

			for (i in 0 until vector[0].size) {
				val ip = ipVecVec(matrixToVector(s, i), matrixToVector(vector, i))
				if (ip < 0f) {
					lambda[i] *= -1f
					for (j in 0 until vector.size) {
						vector[j][i] *= -1f
					}
				}
			}

			return eigSort(lambda, vector)
		}

		// ベクトルを行列形式に
		/**
		 * Converts a vector to a matrix format.
		 * @param v Vector to be converted
		 * @return Matrix representation of the vector
		 */
		fun vectorToMatrix(v: FloatArray): Array<FloatArray> {
			val ans: Array<FloatArray> = Array(v.size) { FloatArray(1) }

			for (i in 0 until v.size) {
				ans[i][0] = v[i]
			}

			return ans
		}

		// 行列をベクトル形式に
		/**
		 * Converts a column of a matrix to a vector.
		 * @param m Matrix
		 * @param n Column index to be converted
		 * @return Vector representation of the specified column
		 */
		fun matrixToVector(m: Array<FloatArray>, n: Int): FloatArray {
			val ans = FloatArray(m.size)

			for (i in 0 until m.size) {
				ans[i] = m[i][n]
			}

			return ans
		}

		// ベクトル表示(文字列付き)
		/**
		 * Prints a vector with a label.
		 * @param str Label for the vector
		 * @param vec Vector to be printed
		 */
		fun printVec(str: String, vec: FloatArray) {
			printMat(str, vectorToMatrix(vec))
		}

		// 行列表示(文字列付き)
		/**
		 * Prints a matrix with a label.
		 * @param str Label for the matrix
		 * @param matrix Matrix to be printed
		 */
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
			for (i in 0 until str.length) {
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
		/**
		 * Prints a vector without a label.
		 * @param vec Vector to be printed
		 */
		fun printVec(vec: FloatArray) {
			printVec("", vec)
		}

		// 行列表示(文字列なし)
		/**
		 * Prints a matrix without a label.
		 * @param matrix Matrix to be printed
		 */
		fun printMat(matrix: Array<FloatArray>) {
			printMat("", matrix)
		}

		// 固有値と固有ベクトルを大きい方から小さい方にソート
		/**
		 * Sorts eigenvalues and corresponding eigenvectors in descending order.
		 * @param lambda Eigenvalues
		 * @param vector Eigenvectors
		 * @return A Pair containing sorted eigenvalues and corresponding eigenvectors
		 */
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