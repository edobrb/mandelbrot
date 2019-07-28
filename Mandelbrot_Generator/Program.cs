using System;
using System.Collections.Generic;
using System.Drawing.Imaging;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Runtime.Serialization;
using System.Drawing;

namespace Mandelbrot_Generator
{
    class Program
    {
        static void Main(string[] args)
        {
            Image image = Bitmap.FromFile(
                @"C:\Users\Edo\Desktop\Mandelbrot\Mandelbrot_View\bin\Windows\x86\Debug\screenshot\8.png");
            PropertyItem p = image.GetPropertyItem(0x010E);
            Console.WriteLine("");
            /*
            int res_y = 100000;
            int res_x = res_y;

            int split_y_gpu = 1;
            int split_y = 2000;
            int maxiter = 100;

            double x = -1;
            double y = 0;
            double viewport = 1.15;

            byte[] colors = new byte[(maxiter + 1) * 4];
            for (int i = 0; i < maxiter + 1; i++)
            {
                byte[] color = MandelbrotHelper.GetLinearGradient(i, 0, maxiter,
                    new MyColor[] { MyColor.DarkGray, MyColor.DarkGray, MyColor.Black,
                        MyColor.Red, MyColor.DarkRed, MyColor.Black },
                    new double[] { 2, 3, 3, 3, 0.5 });

                colors[i * 4 + 0] = color[0];
                colors[i * 4 + 1] = color[1];
                colors[i * 4 + 2] = color[2];
                colors[i * 4 + 3] = color[3];
            }

            string file = "C:\\Users\\Edo\\Desktop\\lol.ppm";

            file = "\\\\192.168.1.10\\Torrent\\lol.ppm";
            if (File.Exists(file)) File.Delete(file);
            BinaryWriter writer = new BinaryWriter(File.OpenWrite(file), Encoding.Default);
            writer.Write(("P6\n" + res_x + " " + res_y + "\n255\n").ToCharArray().Select(c => (byte)c).ToArray());

            MandelbrotGPU gpu = new MandelbrotGPU(2,res_x, res_y / split_y, split_y_gpu);
            gpu.SetNewColors(colors);

            double vx = viewport * res_x / res_y;
            double y_size = viewport * 2 / split_y;
            byte[] buffer = new byte[res_x * res_y / split_y * 3];
            for (int i = 0; i < split_y; i++)
            {
                Console.Write("Drawing " + i + " of " + split_y + "...");
                byte[] data = MandelbrotHelper.GetMandelbrotRegionGPUSplittedY(
                    gpu,
                    x,
                    y,
                    viewport,
                    maxiter,
                    res_x,
                    res_y,
                    split_y,
                    i);
                
                for (int k = 0; k < buffer.Length / 3; k+=1)
                {
                    buffer[k * 3 + 0] = data[k * 4 + 1];
                    buffer[k * 3 + 1] = data[k * 4 + 2];
                    buffer[k * 3 + 2] = data[k * 4 + 3];
                   
                }
                writer.Write(buffer);
                Console.WriteLine(" ok");
            }
            gpu.Close();
            writer.Close();*/
        }
    }
}
