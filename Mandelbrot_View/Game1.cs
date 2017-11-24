using Cudafy.Host;
using Mandelbrot_Generator;
using Microsoft.Xna.Framework;
using Microsoft.Xna.Framework.Graphics;
using Microsoft.Xna.Framework.Input;
using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.IO;
using System.Threading;
using System.Threading.Tasks;

namespace Mandelbrot_View
{
    /// <summary>
    /// This is the main type for your game.
    /// </summary>
    public class Game1 : Game
    {
        GraphicsDeviceManager graphics;
        SpriteBatch spriteBatch;

        int maxiter;

        uint[] colors;
        uint[] buffer;
        bool toUpdate;

        readonly object locker = new object();
        double x, y, viewport;
        double rendered_x, rendered_y, rendered_viewport;

        Texture2D screen, whitePixel, backScreen;
        Settings settings;
        MandelbrotMultiGPU gpu;
        SpriteFont font;

        double maxiter_def;
        KeyboardState oldkey;
        MouseState oldmouse;
        bool showInfo = false;
        TimeSpan renderTime = TimeSpan.FromSeconds(1);
        string screenshotFile = "none";
        public Game1()
        {
            if (File.Exists("settings.json"))
            {
                var json = File.ReadAllText("settings.json");
                settings = JsonConvert.DeserializeObject<Settings>(json);
            }
            else
            {
                throw new Exception("Configuration file not found");
            }


            graphics = new GraphicsDeviceManager(this);

            Content.RootDirectory = "Content";
            this.graphics.PreferredBackBufferWidth = settings.ResolutionX;
            this.graphics.PreferredBackBufferHeight = settings.ResolutionY;
            this.TargetElapsedTime = TimeSpan.FromSeconds(1.0 / 60.0);
            this.graphics.IsFullScreen = settings.FullScreen;
            this.IsMouseVisible = true;

        }


        protected override void Initialize()
        {
            toUpdate = true;
            
            x = 0.261750402076009;
            y = 0.0020502675513626;
            viewport = 1.15;
            maxiter_def = 100;
            /*
            x = 0.261750341770215;
            y = 0.00205035528290217;
            viewport = 1.66 * Math.Pow(10,-12);
            maxiter_def = 30000;*/

            oldkey = Keyboard.GetState();
            base.Initialize();
        }
        protected override void LoadContent()
        {
            spriteBatch = new SpriteBatch(GraphicsDevice);
            screen = new Texture2D(this.GraphicsDevice, settings.RenderResolutionX, settings.RenderResolutionY, false, SurfaceFormat.Color);
            backScreen = new Texture2D(this.GraphicsDevice, settings.ResolutionX, settings.ResolutionY);

            Color g1 = new Color((byte)148, 148, 148);
            Color g2 = new Color((byte)100, 100, 100);
            Color[] gridColor = new Color[settings.ResolutionX * settings.ResolutionY];

            for (int y = 0; y < settings.ResolutionY / 8; y++)
            {
                for (int x = 0; x < settings.ResolutionX / 8; x++)
                {
                    for (int x1 = 0; x1 < 8; x1++)
                        for (int y1 = 0; y1 < 8; y1++)
                            gridColor[(y * 8 + y1) * settings.ResolutionX + (x * 8 + x1)] = y % 2 == 0 ? (x % 2 == 0 ? g1 : g2) : (x % 2 == 0 ? g2 : g1);
                }
            }
            backScreen.SetData<Color>(gridColor);

            //gpu = new MandelbrotGPU(settings.OpenCL_DeviceID, settings.RenderResolutionX, settings.RenderResolutionY , settings.SplitY, buffer, 0);
            gpu = new MandelbrotMultiGPU(settings.Devices_OpenCL_ID.ToArray(),
                settings.Devices_SplitX.ToArray(),
                settings.Devices_PortionY.ToArray(),
                settings.RenderResolutionX, settings.RenderResolutionY);
            RefreshMaxIter();
            font = Content.Load<SpriteFont>("font");
            whitePixel = new Texture2D(GraphicsDevice, 1, 1);
            whitePixel.SetData<byte>(new byte[] { 255, 255, 255, 255 });



        }



        private void RecalcColor(int maxiter)
        {
            colors = new uint[maxiter + 1];
            for (int i = 0; i < maxiter + 1; i++)
            {
                MyColor c = MandelbrotHelper.GetLinearGradient(i, 0, maxiter, settings.Colors.ToArray(), settings.Weight.ToArray());
                colors[i] = c.GetRGBA();
            }
            gpu.SetNewColors(colors);
        }
        private void RefreshMaxIter()
        {
            if (settings.MaxiterMode == MaxiterMode.Dynamic)
            {
                maxiter = (int)(Math.Sqrt(2 * Math.Sqrt(Math.Abs(1 - Math.Sqrt(5 * settings.ResolutionX / viewport)))) * (maxiter_def / 10));
            }
            else
            {
                maxiter = (int)maxiter_def;
            }
        }


        private void Input(GameTime gameTime)
        {
            KeyboardState key = Keyboard.GetState();
            MouseState mouse = Mouse.GetState();
            double zoom_perc_sec = 1.25;
            double move_perc_sec = 2;
            double maxiter_perc_sec = 0.25;

            if (key.IsKeyDown(Keys.W))
            {
                viewport -= viewport * zoom_perc_sec * gameTime.ElapsedGameTime.TotalSeconds;
                toUpdate = true;
                RefreshMaxIter();
            }
            if (key.IsKeyDown(Keys.S))
            {
                viewport += viewport * zoom_perc_sec * gameTime.ElapsedGameTime.TotalSeconds;
                toUpdate = true;
                RefreshMaxIter();
            }
            if (key.IsKeyDown(Keys.Left))
            {
                x -= viewport * move_perc_sec * gameTime.ElapsedGameTime.TotalSeconds;
                toUpdate = true;
            }
            if (key.IsKeyUp(Keys.Left) && oldkey.IsKeyDown(Keys.Left))
            {
                toUpdate = true;
            }
            if (key.IsKeyDown(Keys.Right))
            {
                x += viewport * move_perc_sec * gameTime.ElapsedGameTime.TotalSeconds;
                toUpdate = true;
            }
            if (key.IsKeyDown(Keys.Up))
            {
                y -= viewport * move_perc_sec * gameTime.ElapsedGameTime.TotalSeconds;
                toUpdate = true;
            }
            if (key.IsKeyDown(Keys.Down))
            {
                y += viewport * move_perc_sec * gameTime.ElapsedGameTime.TotalSeconds;
                toUpdate = true;
            }
            if (key.IsKeyDown(Keys.M))
            {
                maxiter_def += maxiter_def * maxiter_perc_sec * gameTime.ElapsedGameTime.TotalSeconds;
                toUpdate = true;
                RefreshMaxIter();
            }
            if (key.IsKeyDown(Keys.N))
            {
                maxiter_def -= maxiter_def * maxiter_perc_sec * gameTime.ElapsedGameTime.TotalSeconds;
                if (maxiter_def < 10) maxiter_def = 10;
                toUpdate = true;
                RefreshMaxIter();
            }
            if (key.IsKeyDown(Keys.F11) && oldkey.IsKeyUp(Keys.F11))
            {
                graphics.ToggleFullScreen();
            }
            if (key.IsKeyDown(Keys.I) && oldkey.IsKeyUp(Keys.I))
            {
                showInfo = !showInfo;
            }
            if (key.IsKeyDown(Keys.F12) && oldkey.IsKeyUp(Keys.F12))
            {
                if (!Directory.Exists("screenshot")) Directory.CreateDirectory("screenshot");
                int offset = 0;
                while (File.Exists("screenshot\\" + (Directory.GetFiles("screenshot").Length + offset) + ".png"))
                {
                    offset++;
                }
                screenshotFile = (Directory.GetFiles("screenshot").Length + offset) + ".png";
                Stream stream = File.Open("screenshot\\" + (Directory.GetFiles("screenshot").Length + offset) + ".png", FileMode.Create);
                screen.SaveAsPng(stream, settings.ResolutionX * 2, settings.ResolutionY * 2);
                stream.Close();

            }
            if (mouse.LeftButton == ButtonState.Pressed && oldmouse.LeftButton == ButtonState.Released)
            {
                if (mouse.X >= 0 && mouse.X <= settings.ResolutionX && mouse.Y >= 0 && mouse.Y <= settings.ResolutionY)
                {
                    double vx = viewport * settings.RenderResolutionX / settings.RenderResolutionY;

                    double xm = ((double)mouse.X * 2) / settings.ResolutionX - 1;
                    double ym = ((double)mouse.Y * 2) / settings.ResolutionY - 1;


                    x += xm * vx;
                    y += ym * viewport;
                    toUpdate = true;
                }
            }

            if (settings.RenderMode == RenderMode.Fluid || settings.RenderMode == RenderMode.Forced)
            {
                if (toUpdate)
                {
                    toUpdate = false;
                    input_changed = true;
                }
            }
            else if (settings.RenderMode == RenderMode.Manual)
            {
                if (key.IsKeyDown(Keys.Enter) && oldkey.IsKeyUp(Keys.Enter))
                {
                    input_changed = true;
                }
            }




            oldkey = key;
            oldmouse = mouse;
        }

        bool rendering = false;
        private void Render()
        {
            RecalcColor(maxiter_copy);
            DateTime start = DateTime.Now;
            double vx = viewport * settings.RenderResolutionX / settings.RenderResolutionY;
            buffer = gpu.GetArea(x - vx, x + vx, y - viewport, y + viewport, maxiter_copy);
            lock (locker)
            {
                screen.SetData<uint>(buffer);
                rendered_x = x_copy;
                rendered_y = y_copy;
                rendered_viewport = viewport_copy;
            }
            renderTime = DateTime.Now - start;




            rendering = false;
        }
        protected override void OnExiting(object sender, EventArgs args)
        {
            if (th != null)
                th.Join();
            gpu.Close();
        }
        Thread th;
        bool input_changed = false;
        double x_copy, y_copy, viewport_copy;
        int maxiter_copy;
        protected override void Draw(GameTime gameTime)
        {
            Input(gameTime);
            if (settings.RenderMode == RenderMode.Forced)
            {
                if (input_changed && !rendering)
                {
                    maxiter_copy = maxiter;
                    x_copy = x;
                    y_copy = y;
                    viewport_copy = viewport;
                    rendering = true;
                    input_changed = false;
                    Render();
                }
            }
            else
            {
                if (input_changed && !rendering)
                {
                    maxiter_copy = maxiter;
                    x_copy = x;
                    y_copy = y;
                    viewport_copy = viewport;
                    rendering = true;
                    input_changed = false;
                    th = new Thread(Render);
                    th.Start();
                }
            }



            lock (locker)
            {
                GraphicsDevice.Clear(Color.Black);
                spriteBatch.Begin();
                var r = new Rectangle(0, 0, settings.ResolutionX, settings.ResolutionY);

                spriteBatch.Draw(backScreen, r, Color.White);



                double vx = viewport * settings.RenderResolutionX / settings.RenderResolutionY;
                double pixel_value = rendered_viewport / settings.RenderResolutionY;

                double center_x = (settings.RenderResolutionX / 2.0) + (x - rendered_x) / pixel_value / 2.0;
                double center_y = (settings.RenderResolutionY / 2.0) + (y - rendered_y) / pixel_value / 2.0;
                double vieport_scale = viewport / rendered_viewport;

                var source_rect = new Rectangle(
                    (int)(center_x - vieport_scale * settings.RenderResolutionX / 2.0),
                    (int)(center_y - vieport_scale * settings.RenderResolutionY / 2.0),
                    (int)(vieport_scale * settings.RenderResolutionX),
                    (int)(vieport_scale * settings.RenderResolutionY)
                    );

                spriteBatch.Draw(screen, r, source_rect, Color.White);

                if (showInfo)
                {
                    r = new Rectangle(0, 0, 265, 125);
                    spriteBatch.Draw(whitePixel, r, Color.Silver);
                    spriteBatch.DrawString(font, "X: " + x, new Vector2(5, 5), Color.Black);
                    spriteBatch.DrawString(font, "Y: " + y, new Vector2(5, 25), Color.Black);
                    spriteBatch.DrawString(font, "Viewport: " + viewport, new Vector2(5, 45), Color.Black);
                    spriteBatch.DrawString(font, "Maxiter: " + maxiter, new Vector2(5, 65), Color.Black);
                    spriteBatch.DrawString(font, "Fps: " + Math.Round(1.0 / renderTime.TotalSeconds, 3), new Vector2(5, 85), Color.Black);
                    spriteBatch.DrawString(font, "Last screenshot: " + screenshotFile, new Vector2(5, 105), Color.Black);
                }
                spriteBatch.End();
                base.Draw(gameTime);
            }
        }
    }
}
