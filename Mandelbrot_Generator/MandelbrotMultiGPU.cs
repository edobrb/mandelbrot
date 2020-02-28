using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace Mandelbrot_Generator
{
    public class MandelbrotMultiGPU
    {
        MandelbrotGPU[] gpus;
        uint[] buffer;
        int[] portions_y;

        public MandelbrotMultiGPU(int[] device_id, int[] split_x, int[] maxiter_per_step, int[] portions_y, int res_x, int res_y)
        {
            this.portions_y = portions_y;
            buffer = new uint[res_x * res_y];
            gpus = new MandelbrotGPU[device_id.Length];
            for (int i = 0; i < gpus.Length; i++)
            {
                int real_res_y = res_y * portions_y[i] / portions_y.Sum();

                int startY = 0;
                for (int k = 0; k < i; k++)
                    startY += res_y * portions_y[k] / portions_y.Sum();

                gpus[i] = new MandelbrotGPU(device_id[i], res_x, real_res_y, split_x[i], buffer, res_x * startY, maxiter_per_step[i]);
            }
        }

        public uint[] GetArea(double x0, double x1, double y0, double y1, int max_iter)
        {
            double y_portion = (y1 - y0) / portions_y.Sum();



            Task[] tasks = new Task[gpus.Length];
            for (int i = 0; i < gpus.Length; i++)
            {
                int my_i = i;
                tasks[i] = new Task(() =>
                {
                    double last_y_end = y0;
                    for (int k = 0; k < my_i; k++) last_y_end += y_portion * portions_y[k];

                    double y0_gpu = last_y_end;
                    double y1_gpu = last_y_end + y_portion * portions_y[my_i];


                    gpus[my_i].UpdateArea(x0, x1, y0_gpu, y1_gpu, max_iter);

                });
            }
            for (int i = 0; i < gpus.Length; i++)
            {
                tasks[i].Start();
            }
            for (int i = 0; i < gpus.Length; i++)
            {
                tasks[i].Wait();
                gpus[i].Syncronize();
            }

            return buffer;
        }
        public void SetNewColors(uint[] colors)
        {
            for (int i = 0; i < gpus.Length; i++)
            {
                gpus[i].SetNewColors(colors);
            }
        }
        public void Close()
        {
            for (int i = 0; i < gpus.Length; i++)
            {
                gpus[i].Close();
            }
        }
    }
}
