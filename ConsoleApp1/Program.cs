using Accord.Math;
using Accord.Statistics.Analysis;
using Cloud.Unum.USearch;
using System.Diagnostics;
using System.Drawing;
using Accord.Statistics.Kernels;
using Accord.Statistics;
using Accord.Math;



Console.WriteLine("USearch");

try
{
    Stopwatch stopwatch = new Stopwatch();
    Stopwatch stopwatchTotal = new Stopwatch();

    stopwatch.Start();
    stopwatchTotal.Start();

    // Paths to images to be indexed
    string[] imagePathsToIndex = new string[]
    {
                "D:\\Test\\USearch\\ConsoleApp1\\image1.jpg",
                "D:\\Test\\USearch\\ConsoleApp1\\image2.jpg",
                "D:\\Test\\USearch\\ConsoleApp1\\image3.jpg",
                "D:\\Test\\USearch\\ConsoleApp1\\image4.jpg",
                "D:\\Test\\USearch\\ConsoleApp1\\image6.jpg",
                "D:\\Test\\USearch\\ConsoleApp1\\queryImage3.jpg",
                "D:\\Test\\USearch\\ConsoleApp1\\queryImage.jpg"
    };

    // Load, resize, and preprocess the images to index
    double[][] vectorsToIndex = LoadAndProcessImages(imagePathsToIndex, out int vectorLengthToIndex);
    //double[][] vectorsToIndex = LoadVectorsFromFile("vectors.txt");

    //int vectorLengthToIndex = vectorsToIndex[0].Length;
    Console.WriteLine(vectorLengthToIndex);
    SaveVectorsToFile(vectorsToIndex, "vectors.txt");

    Console.WriteLine($"Extract vectors from images: {stopwatch.ElapsedMilliseconds} ms");
    stopwatch.Stop();

    // Load, resize, and preprocess the query image
    string queryImagePath = "D:\\Test\\USearch\\ConsoleApp1\\queryImage3.jpg";
    if (!File.Exists(queryImagePath))
    {
        Console.WriteLine("Error: Query image file does not exist.");
        return;
    }

    using Bitmap queryBitmap = new Bitmap(queryImagePath);
    using Bitmap resizedQueryBitmap = ResizeImage(queryBitmap, 100, 100); // Resize to the fixed size
    double[] queryVector = ExtractImageFeatures(resizedQueryBitmap);

    stopwatch.Restart();

    // Create and use the USearchIndex
    using var index = new USearchIndex(
        metricKind: MetricKind.Cos,
        quantization: ScalarKind.Float32,
        dimensions: (ulong)vectorLengthToIndex, // Pass the vector length
        connectivity: 16,
        expansionAdd: 128,
        expansionSearch: 64,
        multi: true
    );

    // Add all images' vectors to the index
    for (ulong i = 0; i < (ulong)vectorsToIndex.Length; i++)
    {
        index.Add(i, vectorsToIndex[i]);
    }

    index.Save("index.usearch");

    Console.WriteLine($"Indexed: {stopwatch.ElapsedMilliseconds} ms");
    stopwatch.Stop();
    stopwatch.Restart();

    // Search for the closest matches using the query image's vector
    int matches = index.Search(queryVector, vectorsToIndex.Length, out ulong[] keys, out float[] distances);

    // Print the results
    for (int i = 0; i < matches; i++)
    {
        Console.WriteLine($"Match found: Image ID = {keys[i]}, Distance = {distances[i]}");
    }

    stopwatch.Stop();
    Console.WriteLine($"Time to search for the closest matches: {stopwatch.ElapsedMilliseconds} ms");
    Console.WriteLine($"Total: {stopwatchTotal.ElapsedMilliseconds} ms");
}
catch (ArgumentException ex)
{
    Console.WriteLine("An error occurred while loading the image: " + ex.Message);
}
catch (Exception ex)
{
    Console.WriteLine("An unexpected error occurred: " + ex.Message);
}


static double[][] LoadAndProcessImages(string[] imagePaths, out int vectorLength)
{
    var vectors = new System.Collections.Generic.List<double[]>();

    foreach (var path in imagePaths)
    {
        if (!File.Exists(path))
        {
            Console.WriteLine($"Error: Image file {path} does not exist.");
            continue;
        }

        using Bitmap bitmap = new Bitmap(path);
        using Bitmap resizedBitmap = ResizeImage(bitmap, 100, 100); // Resize to a fixed size
        double[] vector = ExtractImageFeatures(resizedBitmap);
        vectors.Add(vector);
    }

    if (vectors.Count == 0)
    {
        vectorLength = 0;
        return Array.Empty<double[]>();
    }

    vectorLength = vectors[0].Length; // Assuming all vectors have the same length
    return vectors.ToArray();
}

static Bitmap ResizeImage(Bitmap originalImage, int width, int height)
{
    Bitmap resizedImage = new Bitmap(width, height);
    using (Graphics graphics = Graphics.FromImage(resizedImage))
    {
        graphics.DrawImage(originalImage, 0, 0, width, height);
    }
    return resizedImage;
}

//static double[] ExtractImageFeatures(Bitmap bitmap)
//{
//    int width = bitmap.Width;
//    int height = bitmap.Height;
//    int dimension = width * height * 3; // Assuming RGB image
//    double[] vector = new double[dimension];

//    int index = 0;
//    for (int y = 0; y < height; y++)
//    {
//        for (int x = 0; x < width; x++)
//        {
//            Color pixel = bitmap.GetPixel(x, y);
//            vector[index++] = pixel.R / 255f;
//            vector[index++] = pixel.G / 255f;
//            vector[index++] = pixel.B / 255f;
//        }
//    }

//    return ReduceVectorLength(vector, 128);
//}

static void SaveVectorsToFile(double[][] vectors, string filePath)
{
    using (StreamWriter writer = new StreamWriter(filePath))
    {
        foreach (var vector in vectors)
        {
            string line = string.Join(",", vector);
            writer.WriteLine(line);
        }
    }
}

static double[][] LoadVectorsFromFile(string filePath)
{
    var vectors = new System.Collections.Generic.List<double[]>();

    foreach (var line in File.ReadLines(filePath))
    {
        double[] vector = line.Split(',').Select(double.Parse).ToArray();
        vectors.Add(vector);
    }

    return vectors.ToArray();
}





static double[] ExtractImageFeatures(Bitmap bitmap)
{
    int width = bitmap.Width;
    int height = bitmap.Height;
    int dimension = width * height * 3; // RGB image
    double[] vector = new double[dimension];

    int index = 0;
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            Color pixel = bitmap.GetPixel(x, y);
            vector[index++] = pixel.R / 255.0;
            vector[index++] = pixel.G / 255.0;
            vector[index++] = pixel.B / 255.0;
        }
    }

    // Normalize the vector (Optional step, if needed to match your range)
    NormalizeVector(vector);

    return ReduceVectorLength(vector, 128);
}

// Normalize the vector
static void NormalizeVector(double[] vector)
{
    double mean = 0;
    double variance = 0;

    foreach (double val in vector)
    {
        mean += val;
    }
    mean /= vector.Length;

    foreach (double val in vector)
    {
        variance += (val - mean) * (val - mean);
    }
    variance /= vector.Length;

    double stdDev = Math.Sqrt(variance);

    for (int i = 0; i < vector.Length; i++)
    {
        vector[i] = (vector[i] - mean) / stdDev;
    }
}

// Placeholder function for vector reduction
static double[] ReduceVectorLength(double[] vector, int targetDimension)
{
    // Example approach: Simple average pooling or dimensionality reduction method
    int originalDimension = vector.Length;
    double[] reducedVector = new double[targetDimension];
    int chunkSize = originalDimension / targetDimension;

    for (int i = 0; i < targetDimension; i++)
    {
        double sum = 0;
        for (int j = 0; j < chunkSize; j++)
        {
            sum += vector[i * chunkSize + j];
        }
        reducedVector[i] = sum / chunkSize;
    }

    return reducedVector;
}