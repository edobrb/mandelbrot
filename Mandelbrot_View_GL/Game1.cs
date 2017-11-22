using Cudafy.Host;
using Mandelbrot_Generator;
using Microsoft.Xna.Framework;
using Microsoft.Xna.Framework.Graphics;
using Microsoft.Xna.Framework.Input;
using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.IO;

namespace Mandelbrot_View_GL
{
    /// <summary>
    /// This is the main type for your game.
    /// </summary>
    public class Game1 : Game
    {
        GraphicsDeviceManager graphics;
        SpriteBatch spriteBatch;

        int maxiter = 100;

        uint[] colors;
        bool toUpdate;
        double x, y, viewport;

        Texture2D screen, whitePixel;
        Settings settings;
        MandelbrotGPU gpu;
        SpriteFont font;
        public Game1()
        {
            if (File.Exists("settings.json"))
            {
                var json = File.ReadAllText("settings.json");
                settings = JsonConvert.DeserializeObject<Settings>(json);
            }
            else
            {
                settings = new Settings()
                {
                    ResolutionX = 1920 / 2,
                    ResolutionY = 1080 / 2,
                    SplitY = 2,
                    FullScreen = false,
                    Colors = new List<MyColor>(new MyColor[] { MyColor.DarkGray, MyColor.DarkGray, MyColor.Black,
                        MyColor.Red, MyColor.DarkRed, MyColor.Black }),
                    Weight = new List<double>(new double[] { 1, 1, 1, 1, 1 }),
                    OpenCL_DeviceID = 2,
                };
            }


            graphics = new GraphicsDeviceManager(this);

            Content.RootDirectory = "Content";
            this.graphics.PreferredBackBufferWidth = settings.ResolutionX;
            this.graphics.PreferredBackBufferHeight = settings.ResolutionY;
            this.TargetElapsedTime = TimeSpan.FromSeconds(1.0 / 60.0);
            this.graphics.IsFullScreen = settings.FullScreen;

            this.Window.Position = new Point(1, 1);
        }


        protected override void Initialize()
        {
            toUpdate = true;
            x = -1;
            y = 0;
            viewport = 1.15;


            oldkey = Keyboard.GetState();
            base.Initialize();
        }
        protected override void LoadContent()
        {
            spriteBatch = new SpriteBatch(GraphicsDevice);
            screen = new Texture2D(this.GraphicsDevice, settings.ResolutionX, settings.ResolutionY, false, SurfaceFormat.Color);
            gpu = new MandelbrotGPU(settings.OpenCL_DeviceID, settings.ResolutionX, settings.ResolutionY, settings.SplitY);
            RefreshMaxIter();
            font = Content.Load<SpriteFont>("font");
            whitePixel = new Texture2D(GraphicsDevice, 1, 1);
            whitePixel.SetData<byte>(new byte[] { 255, 255, 255, 255 });
        }


        double maxiter_def = 100;
        KeyboardState oldkey;
        private void RecalcColor()
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
            if (settings.MaxiterMode == "dynamic")
            {
                maxiter = (int)(Math.Sqrt(2 * Math.Sqrt(Math.Abs(1 - Math.Sqrt(5 * settings.ResolutionX / viewport)))) * (maxiter_def / 10));
            }
            else
            {
                maxiter = (int)maxiter_def;
            }
            RecalcColor();
        }
        private void Input(GameTime gameTime)
        {
            KeyboardState key = Keyboard.GetState();
            double zoom_perc_sec = 1.25;
            double move_perc_sec = 2;
            double maxiter_perc_sec = 0.25;

            if (key.IsKeyDown(Keys.W))
            {
                viewport -= viewport * zoom_perc_sec * gameTime.ElapsedGameTime.TotalSeconds;
                RefreshMaxIter();
                toUpdate = true;
            }
            if (key.IsKeyDown(Keys.S))
            {
                viewport += viewport * zoom_perc_sec * gameTime.ElapsedGameTime.TotalSeconds;
                RefreshMaxIter();
                toUpdate = true;
            }

            if (key.IsKeyDown(Keys.Left))
            {
                x -= viewport * move_perc_sec * gameTime.ElapsedGameTime.TotalSeconds;
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
                RefreshMaxIter();
                toUpdate = true;
            }
            if (key.IsKeyDown(Keys.N))
            {
                maxiter_def -= maxiter_def * maxiter_perc_sec * gameTime.ElapsedGameTime.TotalSeconds;
                if (maxiter_def < 10) maxiter_def = 10;
                RefreshMaxIter();
                toUpdate = true;
            }

            if (key.IsKeyDown(Keys.F11) && oldkey.IsKeyUp(Keys.F11))
            {
                graphics.PreferredBackBufferWidth = 1920;
                graphics.PreferredBackBufferHeight = 1080;
                graphics.ToggleFullScreen();
            }
            oldkey = key;
        }

        bool drawed = true;
        protected override void Draw(GameTime gameTime)
        {
            Input(gameTime);
            if (toUpdate)
            {
                toUpdate = false;

                double vx = viewport * settings.ResolutionX / settings.ResolutionY;
                uint[] data = gpu.GetArea(x - vx, x + vx, y - viewport, y + viewport, maxiter);
                screen.SetData<uint>(data);
            }



            GraphicsDevice.Clear(Color.Black);
            spriteBatch.Begin();
            var r = new Rectangle(0, 0, settings.ResolutionX, settings.ResolutionY);
            spriteBatch.Draw(screen, r, Color.White);

            r = new Rectangle(0, 0, 240, 80);
            spriteBatch.Draw(whitePixel, r, Color.White);
            spriteBatch.DrawString(font, "X: " + x, new Vector2(0, 0), Color.Black);
            spriteBatch.DrawString(font, "Y: " + y, new Vector2(0, 20), Color.Black);
            spriteBatch.DrawString(font, "Viewport: " + viewport, new Vector2(0, 40), Color.Black);
            spriteBatch.DrawString(font, "Maxiter: " + maxiter, new Vector2(0, 60), Color.Black);
            spriteBatch.End();
            base.Draw(gameTime);
        }
    }
}
