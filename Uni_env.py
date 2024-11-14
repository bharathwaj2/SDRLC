import math
from typing import Optional, Union

import numpy as np
from numpy import sin,cos,pi

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
        self.x_threshold = 8.0
        self.episode_steps = 0

        self.R0 = 1.5
        self.vel = 1.5
        self.k = 5
        self.lamda = 1

        self.d_goal = 0.4
        self.d_min = 0.75

        # Vehicle state
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0

        # Goal state
        self.xgoal = 0.0
        self.ygoal = 0.0
        self.thetagoal = 0.0

        # Obstacle state
        self.xobs = 0.0
        self.yobs = 0.0
        self.thetaobs = 0.0

        # Observation space for agent
        self.xdel = self.xgoal - self.x
        self.ydel = self.ygoal - self.y
        self.thetadel = self.thetagoal - self.theta

        self.xdel_obs = self.xobs - self.x
        self.ydel_obs = self.yobs - self.y
        self.thetadel_obs = self.thetaobs - self.theta

        self.state = np.array([self.xdel,self.ydel,self.thetadel,self.xdel_obs,self.ydel_obs,self.thetadel_obs]).astype(np.float32)

        high = np.array(
        [
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


        dx,dy,dthe,dxo,dyo,dtheo = self.state
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
        
        dxo = self.xobs - x
        dyo = self.yobs - y
        self.thetaobs = np.arctan2(dyo,dxo)
        self.thetaobs = np.arctan2(sin(self.thetaobs),cos(self.thetaobs))
        dtheo = self.thetaobs - the

        d_diff = (dthe)**2
        d_goal = np.sqrt((dx)**2 + (dy)**2)
        d_obs = np.sqrt((dxo)**2 + (dyo)**2)
        
        bound_terminate = bool(
            x < -self.x_threshold
            or x > self.x_threshold
            or y < -self.x_threshold
            or y > self.x_threshold
        )

        goal_terminate = bool(d_goal < self.d_goal)
        collision_terminate = bool(d_obs < self.d_min)

        truncate = bool(self.episode_steps >= 1000)
        terminate = bound_terminate or collision_terminate
        
        done = terminate or truncate or goal_terminate

        # reward
        if not done:
            reward = -6*(d_diff) - 1.2*(d_goal**2)
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

        self.state = np.array([dx,dy,dthe,dxo,dyo,dtheo],dtype=np.float32)

        obs = self.state

        return obs, reward, done, self.episode_steps, reason

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)
        
        # Agent
        self.x = 0 + np.random.uniform(-1,1,1)[0]
        self.y = 0 + np.random.uniform(-1,1,1)[0]
        self.theta = pi/4
        
        # Goal
        self.xgoal = 6 + np.random.uniform(-1,1,1)[0]
        self.ygoal = 6 + np.random.uniform(-1,1,1)[0]
        self.thetagoal = np.arctan2((self.ygoal-self.y),(self.xgoal-self.x))

        # Obstacle
        self.xobs = 3.0
        self.yobs = 3.0
        self.thetaobs = np.arctan2((self.yobs-self.y),(self.xobs-self.x))

        self.xdel = self.xgoal - self.x
        self.ydel = self.ygoal - self.y
        self.thetadel = self.thetagoal - self.theta

        self.xdel_obs = self.xobs - self.x
        self.ydel_obs = self.yobs - self.y
        self.thetadel_obs = self.thetaobs - self.theta

        self.state = np.array([self.xdel,self.ydel,self.thetadel,self.xdel_obs,self.ydel_obs,self.thetadel_obs]).astype(np.float32)

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

        cartwidth = 20.0
        cartheight = 20.0

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


        obsx1 =  (self.xobs) * scalex + self.screen_width / 2.0 
        obsy1 =  (self.yobs) * scaley + self.screen_height / 2.0 
        gfxdraw.filled_circle(
            self.surf,
            int(obsx1),
            int(obsy1),
            15,
            (0, 0, 255),
        )

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

        return x,y,self.xgoal,self.ygoal,self.xobs,self.yobs
    
    def optimal_control_casadi(self,u_rl):
        
        x = self.xgoal - self.state[0]
        y = self.ygoal - self.state[1]
        the = self.thetagoal - self.state[2]

        # Safety barrier 
        h = (x-self.xobs)**2 + (y-self.yobs)**2 - self.R0**2
        h_dot = 2*(x-self.xobs)*self.vel*cos(the) + 2*(y-self.yobs)*self.vel*sin(the)

        # Lie Derivatives
        L2fh = 2*self.vel**2 
        LgLfh = -2*(x-self.xobs)*self.vel*sin(the) + 2*(y-self.yobs)*self.vel*cos(the)
        
        opti = ca.Opti()
        u = opti.variable()

        opti.minimize(0.5*(dot((u-u_rl),(u-u_rl))))

        # Constraint
        opti.subject_to(dot(LgLfh,u) + L2fh + self.lamda*(h_dot + self.k*h) >= 0)

        option = {"verbose": False, "ipopt.print_level": 0, "print_time": 0}
        opti.solver('ipopt',option)

        sol = opti.solve()
        u_opt = sol.value(u)

        return u_opt
    
