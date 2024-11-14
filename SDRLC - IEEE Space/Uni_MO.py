import math
from typing import Optional, Union

import numpy as np
from numpy import *

import gym
from gym import logger, spaces
# from gym.envs.classic_control import utils
from gym.error import DependencyNotInstalled
import matplotlib.pyplot as plt
import casadi as ca
from casadi import *



class Unicycle(gym.Env):

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 50,
    }

    def __init__(self, render_mode: Optional[str] = None, method=None):
        super(Unicycle, self).__init__()

        self.dt = 0.01
        self.x_threshold = 12
        self.episode_steps = 0

        self.R0 = 1.5
        self.vel = 1.5
        self.k = 5
        self.lamda = 1

        self.d_goal = 0.6
        self.d_min = 1.0

        # Vehicle state
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0

        # Goal state
        self.xgoal = 0.0
        self.ygoal = 0.0
        self.thetagoal = 0.0

        # Obstacle state
        self.xobs_list = []
        self.yobs_list = []
        self.thetaobs_list = []

        # Observation space for agent
        self.xdel = 0.0
        self.ydel = 0.0
        self.thetadel = 0.0

        self.xdel_obs1 = 0.0
        self.ydel_obs1 = 0.0
        self.thetadel_obs1 = 0.0

        self.xdel_obs2 = 0.0
        self.ydel_obs2 = 0.0
        self.thetadel_obs2 = 0.0

        self.xdel_obs3 = 0.0
        self.ydel_obs3 = 0.0
        self.thetadel_obs3 = 0.0

        self.state = np.array([self.xdel,self.ydel,self.thetadel,self.xdel_obs1,self.ydel_obs1,self.thetadel_obs1,
                               self.xdel_obs2,self.ydel_obs2,self.thetadel_obs2,self.xdel_obs3,self.ydel_obs3,self.thetadel_obs3]).astype(np.float32)
        
        high = np.array(
        [
            self.x_threshold * 2,
            self.x_threshold * 2,
            pi,
            self.x_threshold * 2,
            self.x_threshold * 2,
            pi,
            self.x_threshold * 2,
            self.x_threshold * 2,
            pi,
            self.x_threshold * 2,
            self.x_threshold * 2,
            pi

        ],dtype=np.float32)
        
        # Define action and observation space
        self.action_space = spaces.Box(-2*pi, +2*pi, (1,), dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        self.render_mode = render_mode

        self.screen_width = 500
        self.screen_height = 500
        self.screen = None
        self.clock = None
        self.isopen = True
        self.state = None
        self.method = method

        self.steps_beyond_terminated = None

    def step(self, action):

        reason = 'nothing'
        assert self.state is not None, "Call reset before using step method."

        action = np.clip(action, -4.0, +4.0)

        if self.method == 'RL-CBF':
            omega = self.optimal_control_casadi(action)
        else:
            omega = action

        dx,dy,dthe,dxo1,dyo1,dtheo1,dxo2,dyo2,dtheo2,dxo3,dyo3,dtheo3 = self.state
        x = self.xgoal - dx 
        y = self.ygoal - dy
        the = self.thetagoal - dthe

        # dynamics
        x += self.dt * self.vel*cos(the)
        y += self.dt * self.vel*sin(the)
        the += self.dt * omega
        
        the = np.arctan2(sin(the),cos(the))
        
        dx = self.xgoal - x
        dy = self.ygoal - y
        self.thetagoal = np.arctan2(dy,dx)
        self.thetagoal = np.arctan2(sin(self.thetagoal),cos(self.thetagoal))
        dthe = self.thetagoal - the

        self.updata_obs_theta()
        
        dxo1 = self.xobs_list[0] - x
        dyo1 = self.yobs_list[0] - y
        dtheo1 = self.thetaobs_list[0] - the

        dxo2 = self.xobs_list[1] - x
        dyo2 = self.yobs_list[1] - y
        dtheo2 = self.thetaobs_list[1] - the

        dxo3 = self.xobs_list[2] - x
        dyo3 = self.yobs_list[2] - y
        dtheo3 = self.thetaobs_list[2] - the

        d_diff = (dthe)**2
        d_goal = np.sqrt((dx)**2 + (dy)**2)

        d_obs1 = np.sqrt((dxo1)**2 + (dyo1)**2)
        d_obs2 = np.sqrt((dxo2)**2 + (dyo2)**2)
        d_obs3 = np.sqrt((dxo3)**2 + (dyo3)**2)
        
        bound_terminate = bool(
            x < -self.x_threshold
            or x > self.x_threshold
            or y < -self.x_threshold
            or y > self.x_threshold
        )

        goal_terminate = bool(d_goal < self.d_goal)
        collision_terminate = bool((d_obs1 < self.d_min) or (d_obs2 < self.d_min) or (d_obs3 < self.d_min))

        truncate = bool(self.episode_steps >= 1800)
        terminate = bound_terminate or collision_terminate
        
        done = terminate or truncate or goal_terminate

        # reward
        if not done:
            reward = -6*(d_diff) - 3*(d_goal) - 1
            self.episode_steps+=1
            reason = 'working fine'
        elif self.steps_beyond_terminated is None:
            reward = 0.0
            if bound_terminate :
                reward = -10000.0
                reason = 'out of bounds'
            elif collision_terminate:
                reward = -10000.0
                reason = 'collision'
            elif goal_terminate:
                reward = 20000.0
                reason = 'goal reached'
            elif truncate and not goal_terminate:
                reward = -1000.0
                reason = 'max steps'

            self.steps_beyond_terminated = 0
        else:
            self.steps_beyond_terminated += 1

        self.state = np.array([dx,dy,dthe,dxo1,dyo1,dtheo1,dxo2,dyo2,dtheo2,dxo3,dyo3,dtheo3],dtype=np.float32)

        obs = self.state

        return obs, reward, done, self.episode_steps, reason

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)
        
        # Agent
        self.x = -8 + np.random.uniform(-1,1,1)[0]
        self.y = -7 + np.random.uniform(-1,1,1)[0]
        self.theta = pi/4
        
        # Goal
        self.xgoal = 8 + np.random.uniform(-1,1,1)[0]
        self.ygoal = 7 + np.random.uniform(-1,1,1)[0] 
        self.thetagoal = np.arctan2((self.ygoal-self.y),(self.xgoal-self.x))

        # Obstacle
        angle1 = deg2rad(255)
        angle2 = deg2rad(195)
        self.xobs_list = [2,2+(4*cos(angle1)),2+(4*cos(angle2))] 
        self.yobs_list = [4,4+(4*sin(angle1)),4+(4*sin(angle2))]

        self.thetaobs_list = [0,0,0]
        self.updata_obs_theta()

        self.xdel = self.xgoal - self.x
        self.ydel = self.ygoal - self.y
        self.thetadel = self.thetagoal - self.theta

        self.xdel_obs1 = self.xobs_list[0] - self.x
        self.ydel_obs1 = self.yobs_list[0] - self.y
        self.thetadel_obs1 = self.thetaobs_list[0] - self.theta

        self.xdel_obs2 = self.xobs_list[1] - self.x
        self.ydel_obs2 = self.yobs_list[1] - self.y
        self.thetadel_obs2 = self.thetaobs_list[1] - self.theta

        self.xdel_obs3 = self.xobs_list[2] - self.x
        self.ydel_obs3 = self.yobs_list[2] - self.y
        self.thetadel_obs3 = self.thetaobs_list[2] - self.theta


        self.state = np.array([self.xdel,self.ydel,self.thetadel,self.xdel_obs1,self.ydel_obs1,self.thetadel_obs1,
                               self.xdel_obs2,self.ydel_obs2,self.thetadel_obs2,self.xdel_obs3,self.ydel_obs3,self.thetadel_obs3]).astype(np.float32)

        self.steps_beyond_terminated = None
        self.episode_steps = 0

        return self.state

    def render(self):
        if self.render_mode is None:
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym("{self.spec.id}", render_mode="rgb_array")'
            )
            return

        try:
            import pygame
            from pygame import gfxdraw
        except ImportError:
            raise DependencyNotInstalled(
                "pygame is not installed, run `pip install gym[classic_control]`"
            )

        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode(
                    (self.screen_width, self.screen_height)
                )
            else:  # mode == "rgb_array"
                self.screen = pygame.Surface((self.screen_width, self.screen_height))
        if self.clock is None:
            self.clock = pygame.time.Clock()
  

        world_width = self.x_threshold * 2
        world_height = self.x_threshold * 2
        scalex = self.screen_width / world_width
        scaley = self.screen_height / world_height

        cartwidth = 12.0
        cartheight = 12.0

        if self.state is None:
            return None

        x = self.state
        xc = self.xgoal - x[0]
        yc = self.ygoal - x[1]
        # xc = x[0]
        # yc = x[1]
        # rxobs = np.array([x[4],x[5]], dtype=np.float32)
        # ryobs = np.array([x[6],x[7]], dtype=np.float32)

        self.surf = pygame.Surface((self.screen_width, self.screen_height))
        self.surf.fill((255, 255, 255))

        l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2

        cartx = xc * scalex + self.screen_width / 2.0  # MIDDLE OF CART
        carty = yc * scaley + self.screen_height / 2.0  # TOP OF CART

        cart_coords = [(l, b), (l, t), (r, t), (r, b)]
        cart_coords = [(c[0] + cartx, c[1] + carty) for c in cart_coords]
        gfxdraw.aapolygon(self.surf, cart_coords, (0, 0, 0))
        gfxdraw.filled_polygon(self.surf, cart_coords, (0, 0, 0))
        
        goalx = self.xgoal * scalex + self.screen_width / 2.0
        goaly = self.ygoal * scaley + self.screen_height / 2.0
        gfxdraw.filled_circle(
            self.surf,
            int(goalx),
            int(goaly),
            5,
            (255, 0, 0),
        )


        obsx1 =  (self.xobs_list[0]) * scalex + self.screen_width / 2.0 
        obsy1 =  (self.yobs_list[0]) * scaley + self.screen_height / 2.0 
        gfxdraw.filled_circle(
            self.surf,
            int(obsx1),
            int(obsy1),
            10,
            (0, 0, 255),
        )

        obsx1 =  (self.xobs_list[1]) * scalex + self.screen_width / 2.0 
        obsy1 =  (self.yobs_list[1]) * scaley + self.screen_height / 2.0 
        gfxdraw.filled_circle(
            self.surf,
            int(obsx1),
            int(obsy1),
            10,
            (0, 0, 255),
        )

        obsx1 =  (self.xobs_list[2]) * scalex + self.screen_width / 2.0 
        obsy1 =  (self.yobs_list[2]) * scaley + self.screen_height / 2.0 
        gfxdraw.filled_circle(
            self.surf,
            int(obsx1),
            int(obsy1),
            10,
            (0, 0, 255),
        )

        # xc2 = state2[0] - g2[0]
        # yc2 = state2[1] - g2[1]
        # obsx2 =  (xc2) * scalex + self.screen_width / 2.0 
        # obsy2 =  (yc2) * scaley + self.screen_height / 2.0 
        # cart_coords = [(l, b), (l, t), (r, t), (r, b)]
        # cart_coords = [(c[0] + obsx2, c[1] + obsy2) for c in cart_coords]
        # gfxdraw.aapolygon(self.surf, cart_coords, (128, 128, 128))
        # gfxdraw.filled_polygon(self.surf, cart_coords, (128, 128, 128))
        
        # gfxdraw.hline(self.surf, 0, self.screen_width, carty, (0, 0, 0))

        self.font = pygame.font.SysFont("Arial", 20)
        # self.text = self.font.render("Episode: {}    Reward: {:0.1f}    Steps: {} ".format(ep,rw,st), True, (0,0,0)) # {:0.2f}   {:0.2f}   {:0.2f}
        # self.text2 = self.font.render("{:0.2f}   {:0.2f} \t\t.... {:0.2f}   {:0.2f}   {:0.2f}".format((xc),(xobs),(rxobs),(ryobs),self.c), True, (0,0,0))  

        self.surf = pygame.transform.flip(self.surf, False, True)
        self.screen.blit(self.surf, (0, 0))
        # self.screen.blit(self.text, (3, 3))
        # self.screen.blit(self.text2, (40, 40))
        if self.render_mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.update()
            pygame.display.flip()
            

        elif self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )

    def close(self):
        if self.screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.isopen = False

    def plot_trajectory(self):
        
        x = self.xgoal - self.state[0] 
        y = self.ygoal - self.state[1]

        return x,y,self.xgoal,self.ygoal,self.xobs_list,self.yobs_list
    
    def optimal_control_casadi(self,u_rl):
        
        x = self.xgoal - self.state[0]
        y = self.ygoal - self.state[1]
        the = self.thetagoal - self.state[2]

        # Safety barrier 
        h1 = (x-self.xobs_list[0])**2 + (y-self.yobs_list[0])**2 - self.R0**2
        h1_dot = 2*(x-self.xobs_list[0])*self.vel*cos(the) + 2*(y-self.yobs_list[0])*self.vel*sin(the)

        h2 = (x-self.xobs_list[1])**2 + (y-self.yobs_list[1])**2 - self.R0**2
        h2_dot = 2*(x-self.xobs_list[1])*self.vel*cos(the) + 2*(y-self.yobs_list[1])*self.vel*sin(the)

        h3 = (x-self.xobs_list[2])**2 + (y-self.yobs_list[2])**2 - self.R0**2
        h3_dot = 2*(x-self.xobs_list[2])*self.vel*cos(the) + 2*(y-self.yobs_list[2])*self.vel*sin(the)

        # Lie Derivatives
        L2fh = 2*self.vel**2 

        LgLfh1 = -2*(x-self.xobs_list[0])*self.vel*sin(the) + 2*(y-self.yobs_list[0])*self.vel*cos(the)
        LgLfh2 = -2*(x-self.xobs_list[1])*self.vel*sin(the) + 2*(y-self.yobs_list[1])*self.vel*cos(the)
        LgLfh3 = -2*(x-self.xobs_list[2])*self.vel*sin(the) + 2*(y-self.yobs_list[2])*self.vel*cos(the)
        
        opti = ca.Opti()
        u = opti.variable()

        opti.minimize(0.5*(dot((u-u_rl),(u-u_rl))))

        # Constraint
        opti.subject_to(dot(LgLfh1,u) + L2fh + self.lamda*(h1_dot) >= -(self.k)*h1)
        opti.subject_to(dot(LgLfh2,u) + L2fh + self.lamda*(h2_dot) >= -(self.k)*h2)
        opti.subject_to(dot(LgLfh3,u) + L2fh + self.lamda*(h3_dot) >= -(self.k)*h3)

        option = {"verbose": False, "ipopt.print_level": 0, "print_time": 0}
        opti.solver('ipopt',option)

        sol = opti.solve()
        u_opt = sol.value(u)

        return u_opt
    
    def updata_obs_theta(self):
        for i in range(0,3):
            self.thetaobs_list[i] = np.arctan2((self.yobs_list[i]-self.y),(self.xobs_list[i]-self.x))
            self.thetaobs_list[i] = np.arctan2(sin(self.thetaobs_list[i]),cos(self.thetaobs_list[i]))
  