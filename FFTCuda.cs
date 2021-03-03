using ManagedCuda;
using ManagedCuda.CudaFFT;
using ManagedCuda.VectorTypes;
using System;

namespace FFTManagedCUDA
{
    class FFTCuda
    {
        static void Main(string[] args)
        {
            int i;
            var watch = new System.Diagnostics.Stopwatch();
            float[] inputData = new float[1000000];
            Random rand = new Random();
            for (i = 0; i < inputData.Length; i++)
            {
                inputData[i] = (float)rand.NextDouble();
            }
            //for (i = 0; i < inputData.Length; i++)
            //{
            //    Console.WriteLine(inputData[i]);
            //}
            watch.Start();
            float2[] resultFFT =calculateCudaFFT(inputData);
            watch.Stop();
            Console.WriteLine($"Execution Time: {watch.ElapsedMilliseconds} ms");
            Console.WriteLine("Result (FFT): ");
            //for (i = 0; i <resultFFT.Length; i++)
            //{
            //    Console.WriteLine(resultFFT[i]);
            //}
        }
        public static float2[] calculateCudaFFT(float[] h_dataIn)
        {
             CudaContext cntxt = new CudaContext();
            //Caution: Array sizes matter! Based on CUFFFT-Documentation...
            int size_real = h_dataIn.Length;
            int size_complex = (int)Math.Floor(size_real / 2.0) + 1;

            //Crating FFT Plan
            CudaFFTPlanMany fftPlan = new CudaFFTPlanMany(1, new int[] { size_real }, 1, cufftType.R2C);
            
            //Size of d_data must be padded for inplace R2C transforms: size_complex * 2 and not size_real
            CudaDeviceVariable<float> d_data = new CudaDeviceVariable<float>(size_complex * 2);

            //device allocation and host have different sizes, why the amount of data must be given explicitly for copying:
            d_data.CopyToDevice(h_dataIn, 0, 0, size_real * sizeof(float));

            //executa plan
            fftPlan.Exec(d_data.DevicePointer, TransformDirection.Forward);

            //Output to host, either as float2 or float, but array sizes must be right!
            float2[] h_dataOut = new float2[size_complex];
            float[] h_dataOut2 = new float[size_complex * 2];
            d_data.CopyToHost(h_dataOut);
            d_data.CopyToHost(h_dataOut2);
            fftPlan.Dispose();
            return h_dataOut;
        }
        
    }

}
